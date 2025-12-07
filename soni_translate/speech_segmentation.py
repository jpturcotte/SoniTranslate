from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import gc
import os
import soundfile as sf
import nltk
import librosa
import difflib
from tqdm import tqdm
from contextlib import contextmanager
from IPython.utils import capture # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command

ASR_MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "Systran/faster-distil-whisper-large-v3",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "distil-small.en",
    "distil-medium.en",
    "OpenAI_API_Whisper",
    "ibm-granite/granite-speech-3.3-2b",
]

COMPUTE_TYPE_GPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "float16",
    "bfloat16",
    "float32"
]

COMPUTE_TYPE_CPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int16",
    "float32",
]

WHISPER_MODELS_PATH = './WHISPER_MODELS'


@contextmanager
def _force_weights_only_false():
    """Temporarily default ``torch.load`` to ``weights_only=False``.

    WhisperX loads the pyannote VAD model via Lightning, which ultimately calls
    ``torch.load`` without exposing a ``weights_only`` override. Under PyTorch
    2.6 the default flipped to ``True``, which breaks checkpoints containing
    pickled metadata (e.g., OmegaConf objects). By forcing ``weights_only=False``
    only for the duration of model loading, we avoid a global safe-global
    allowlist while keeping the change scoped to this operation.
    """

    original_load = torch.load

    def _load_with_weights_only_false(*args, **kwargs):
        # Always force ``weights_only=False`` even if a caller explicitly
        # requests otherwise, since the pyannote checkpoints rely on pickled
        # metadata that is blocked when ``weights_only=True``.
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _load_with_weights_only_false
    try:
        yield
    finally:
        torch.load = original_load


def openai_api_whisper(
    input_audio_file,
    source_lang=None,
    chunk_duration=1800
):

    info = sf.info(input_audio_file)
    duration = info.duration

    output_directory = "./whisper_api_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    if duration > chunk_duration:
        # Split the audio file into smaller chunks with 30-minute duration
        cm = f'ffmpeg -i "{input_audio_file}" -f segment -segment_time {chunk_duration} -c:a libvorbis "{output_directory}/output%03d.ogg"'
        run_command(cm)
        # Get list of generated chunk files
        chunk_files = sorted(
            [f"{output_directory}/{f}" for f in os.listdir(output_directory) if f.endswith('.ogg')]
        )
    else:
        one_file = f"{output_directory}/output000.ogg"
        cm = f'ffmpeg -i "{input_audio_file}" -c:a libvorbis {one_file}'
        run_command(cm)
        chunk_files = [one_file]

    # Transcript
    segments = []
    language = source_lang if source_lang else None
    for i, chunk in enumerate(chunk_files):
        from openai import OpenAI
        client = OpenAI()

        audio_file = open(chunk, "rb")
        transcription = client.audio.transcriptions.create(
          model="whisper-1",
          file=audio_file,
          language=language,
          response_format="verbose_json",
          timestamp_granularities=["segment"],
        )

        try:
            transcript_dict = transcription.model_dump()
        except: # noqa
            transcript_dict = transcription.to_dict()

        if language is None:
            logger.info(f'Language detected: {transcript_dict["language"]}')
            language = TO_LANGUAGE_CODE[transcript_dict["language"]]

        chunk_time = chunk_duration * (i)

        for seg in transcript_dict["segments"]:

            if "start" in seg.keys():
                segments.append(
                    {
                        "text": seg["text"],
                        "start": seg["start"] + chunk_time,
                        "end": seg["end"] + chunk_time,
                    }
                )

    audio = whisperx.load_audio(input_audio_file)
    result = {"segments": segments, "language": language}

    return audio, result


def find_best_overlap_cutoff(prev_tail, curr_head, search_window=60):
    """Locate fuzzy overlap between previous and current transcription text.

    Parameters
    ----------
    prev_tail: str
        The accumulated transcription text from prior chunks.
    curr_head: str
        The freshly generated transcription for the current chunk.
    search_window: int, optional
        Maximum number of trailing/leading characters to search for overlap.

    Returns
    -------
    int
        The index in ``curr_head`` where unique text begins (i.e., after the
        overlapping portion). ``0`` indicates no overlap found.
    """

    if not prev_tail or not curr_head:
        return 0

    tail = prev_tail[-search_window:]
    head = curr_head[:search_window]

    matcher = difflib.SequenceMatcher(None, tail, head, autojunk=False)
    match = matcher.find_longest_match(0, len(tail), 0, len(head))

    if match.size >= 5:
        return match.b + match.size

    return 0


def granite_speech_transcribe(
    input_audio_file,
    model_id,
    compute_type="float16",
    batch_size=16,
    source_lang=None,
    segment_duration_limit=30,
):
    """Transcribe audio with IBM Granite Speech using direct model calls.

    Implements advanced context handling and safer decoding inspired by the
    research prompt in the user request:

    * Moving Context Window: preserves a short history between chunks.
    * Context Fencing Prompt: separates context from instructions to reduce
      hallucinations.
    * Robust Overlap-Stitch: word-based stitching to handle boundary
      inconsistencies.
    * Safe Decoding: deterministic beam search with repetition penalty.
    """

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import nltk

    # Ensure NLTK data is available for sentence splitting
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Select compute type based on hardware capabilities
    if compute_type == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch_dtype,
        )
        tokenizer = processor.tokenizer
    except Exception as exc:
        logger.error(f"Failed to load Granite Speech model: {exc}")
        raise

    # --- PROMPT CONFIGURATION BASED ON RESEARCH ---
    system_prompt = "You are a precise transcription system."

    # --- CHUNKING WITH OVERLAP ---
    total_duration = sf.info(input_audio_file).duration
    # Enforce minimum chunk size of 5s to avoid micro-segment instability
    chunk_length = float(max(segment_duration_limit, 5))
    chunk_overlap = 1.5  # Slightly larger overlap to ensure boundary continuity

    output_directory = "./granite_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    chunk_files = []
    chunk_times = []  # Store (start, end) for each chunk

    current_start = 0.0
    chunk_idx = 0

    while current_start < total_duration:
        chunk_end = min(current_start + chunk_length, total_duration)
        duration = chunk_end - current_start

        filename = f"{output_directory}/output{chunk_idx:03d}.ogg"

        cm = (
            f'ffmpeg -y -hide_banner -loglevel error -ss {current_start} -t {duration} '
            f'-i "{input_audio_file}" -c:a libvorbis "{filename}"'
        )
        run_command(cm)

        chunk_files.append(filename)
        chunk_times.append((current_start, chunk_end))

        if chunk_end >= total_duration:
            break

        current_start += (chunk_length - chunk_overlap)
        chunk_idx += 1

    segments = []
    language = source_lang or "en"

    progress_bar = tqdm(
        chunk_files,
        desc="Granite chunk processing",
        unit="chunk",
    )

    # Track context and assembled transcript for stitching
    full_transcription_words = []
    previous_context_window = ""

    for i, chunk_path in enumerate(progress_bar):
        try:
            audio_array, _ = librosa.load(chunk_path, sr=16000)

            # --- DYNAMIC PROMPT CONSTRUCTION (CONTEXT FENCING) ---
            if previous_context_window:
                user_content = (
                    f"Context Information: {previous_context_window}\n"
                    "Instruction: Transcribe the following audio exactly as spoken. "
                    "Do not add information from the context if it is not present in the audio.\n"
                    "<|audio|>"
                )
            else:
                user_content = (
                    "Instruction: Transcribe the following audio exactly as spoken.\n"
                    "<|audio|>"
                )

            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            text_prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = processor(
                text_prompt,
                audio_array,
                device=device,
                return_tensors="pt",
            ).to(device)

            # --- GENERATION PARAMETERS (ANTI-HALLUCINATION) ---
            model_outputs = model.generate(
                **model_inputs,
                max_new_tokens=400,
                num_beams=4,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.15,
            )

            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = torch.unsqueeze(
                model_outputs[0, num_input_tokens:],
                dim=0,
            )

            text = tokenizer.batch_decode(
                new_tokens,
                add_special_tokens=False,
                skip_special_tokens=True,
            )[0].strip()

        except Exception as exc:
            logger.error(f"Error processing chunk {chunk_path}: {exc}")
            continue

        if not text:
            continue

        # --- WORD-BASED OVERLAP STITCHING ---
        curr_words = text.split()
        unique_words = curr_words

        if full_transcription_words:
            tail_words = full_transcription_words[-15:]
            head_words = curr_words[:15]

            best_match_len = 0
            for length in range(min(len(tail_words), len(head_words)), 0, -1):
                if tail_words[-length:] == head_words[:length]:
                    best_match_len = length
                    break

            if best_match_len > 0:
                unique_words = curr_words[best_match_len:]

        unique_text = " ".join(unique_words).strip()

        if not unique_text:
            continue

        full_transcription_words.extend(unique_words)

        # --- UPDATE CONTEXT WINDOW ---
        context_len = 25
        if len(full_transcription_words) > context_len:
            previous_context_window = " ".join(full_transcription_words[-context_len:])
        else:
            previous_context_window = " ".join(full_transcription_words)

        # --- SENTENCE SEGMENTATION & TIMESTAMP INTERPOLATION ---
        c_start, c_end = chunk_times[i]

        sentences = nltk.sent_tokenize(unique_text)
        if not sentences:
            sentences = [unique_text]

        lengths = [max(len(s), 1) for s in sentences]
        total_len = sum(lengths)

        cursor = c_start
        if i > 0:
            cursor += chunk_overlap
            cursor = min(cursor, c_end)

        available_duration = max(c_end - cursor, 0.1)

        for s_text, s_len in zip(sentences, lengths):
            proportion = s_len / total_len
            duration = proportion * available_duration

            seg_end = min(cursor + duration, total_duration)
            segments.append({
                "text": s_text,
                "start": cursor,
                "end": seg_end
            })
            cursor = seg_end

    progress_bar.close()

    if not segments:
        segments = [{"text": "", "start": 0.0, "end": total_duration}]

    if segments[-1]["end"] > total_duration:
        segments[-1]["end"] = total_duration

    audio = whisperx.load_audio(input_audio_file)
    return audio, {"segments": segments, "language": language}


def find_whisper_models():
    path = WHISPER_MODELS_PATH
    folders = []

    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if (
                os.path.isdir(folder_path)
                and 'model.bin' in os.listdir(folder_path)
            ):
                folders.append(folder)
    return folders


def transcribe_speech(
    audio_wav,
    asr_model,
    compute_type,
    batch_size,
    SOURCE_LANGUAGE,
    literalize_numbers=True,
    segment_duration_limit=15,
):
    """
    Transcribe speech using a whisper model.

    Parameters:
    - audio_wav (str): Path to the audio file in WAV format.
    - asr_model (str): The whisper model to be loaded.
    - compute_type (str): Type of compute to be used (e.g., 'int8', 'float16').
    - batch_size (int): Batch size for transcription.
    - SOURCE_LANGUAGE (str): Source language for transcription.

    Returns:
    - Tuple containing:
        - audio: Loaded audio file.
        - result: Transcription result as a dictionary.
    """

    if asr_model == "OpenAI_API_Whisper":
        if literalize_numbers:
            logger.info(
                "OpenAI's API Whisper does not support "
                "the literalization of numbers."
            )
        return openai_api_whisper(audio_wav, SOURCE_LANGUAGE)

    if asr_model == "ibm-granite/granite-speech-3.3-2b":
        return granite_speech_transcribe(
            audio_wav,
            asr_model,
            compute_type,
            batch_size,
            SOURCE_LANGUAGE,
            segment_duration_limit,
        )

    # https://github.com/openai/whisper/discussions/277
    prompt = "以下是普通话的句子。" if SOURCE_LANGUAGE == "zh" else None
    SOURCE_LANGUAGE = (
        SOURCE_LANGUAGE if SOURCE_LANGUAGE != "zh-TW" else "zh"
    )
    asr_options = {
        "initial_prompt": prompt,
        "suppress_numerals": literalize_numbers
    }

    if asr_model not in ASR_MODEL_OPTIONS:

        base_dir = WHISPER_MODELS_PATH
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        model_dir = os.path.join(base_dir, sanitize_file_name(asr_model))

        if not os.path.exists(model_dir):
            from ctranslate2.converters import TransformersConverter

            quantization = "float32"
            # Download new model
            try:
                converter = TransformersConverter(
                    asr_model,
                    low_cpu_mem_usage=True,
                    copy_files=[
                        "tokenizer_config.json", "preprocessor_config.json"
                    ]
                )
                converter.convert(
                    model_dir,
                    quantization=quantization,
                    force=False
                )
            except Exception as error:
                if "File tokenizer_config.json does not exist" in str(error):
                    converter._copy_files = [
                        "tokenizer.json", "preprocessor_config.json"
                    ]
                    converter.convert(
                        model_dir,
                        quantization=quantization,
                        force=True
                    )
                else:
                    raise error

        asr_model = model_dir
        logger.info(f"ASR Model: {str(model_dir)}")

    with _force_weights_only_false():
        model = whisperx.load_model(
            asr_model,
            os.environ.get("SONITR_DEVICE"),
            compute_type=compute_type,
            language=SOURCE_LANGUAGE,
            asr_options=asr_options,
        )

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=segment_duration_limit,
        print_progress=True,
    )

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.

    Parameters:
    - audio (array): The audio data in a suitable format for alignment.
    - result (dict): Metadata containing information about the segments
         and language.

    Returns:
    - result (dict): Updated metadata after aligning the segments with
        the audio. This includes character-level alignments if
        'return_char_alignments' is set to True.

    Notes:
    - This function uses language-specific models to align speech segments.
    - It performs language compatibility checks and selects the
        appropriate alignment model.
    - Cleans up memory by releasing resources after alignment.
    """
    DAMHF.update(DAMT)  # lang align
    if (
        not result["language"] in DAMHF.keys()
        and not result["language"] in EXTRA_ALIGN.keys()
    ):
        logger.warning(
            "Automatic detection: Source language not compatible with align"
        )
        raise ValueError(
            f"Detected language {result['language']}  incompatible, "
            "you can select the source language to avoid this error."
        )
    if (
        result["language"] in EXTRA_ALIGN.keys()
        and EXTRA_ALIGN[result["language"]] == ""
    ):
        lang_name = (
            INVERTED_LANGUAGES[result["language"]]
            if result["language"] in INVERTED_LANGUAGES.keys()
            else result["language"]
        )
        logger.warning(
            "No compatible wav2vec2 model found "
            f"for the language '{lang_name}', skipping alignment."
        )
        return result

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=os.environ.get("SONITR_DEVICE"),
        model_name=None
        if result["language"] in DAMHF.keys()
        else EXTRA_ALIGN[result["language"]],
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        os.environ.get("SONITR_DEVICE"),
        return_char_alignments=True,
        print_progress=False,
    )
    del model_a
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "disable": "",
}


def reencode_speakers(result):

    if result["segments"][0]["speaker"] == "SPEAKER_00":
        return result

    speaker_mapping = {}
    counter = 0

    logger.debug("Reencode speakers")

    for segment in result["segments"]:
        old_speaker = segment["speaker"]
        if old_speaker not in speaker_mapping:
            speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
            counter += 1
        segment["speaker"] = speaker_mapping[old_speaker]

    return result


def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization@2.1",
):
    """
    Performs speaker diarization on speech segments.

    Parameters:
    - audio_wav (array): Audio data in WAV format to perform speaker
        diarization.
    - result (dict): Metadata containing information about speech segments
        and alignments.
    - min_speakers (int): Minimum number of speakers expected in the audio.
    - max_speakers (int): Maximum number of speakers expected in the audio.
    - YOUR_HF_TOKEN (str): Your Hugging Face API token for model
        authentication.
    - model_name (str): Name of the speaker diarization model to be used
        (default: "pyannote/speaker-diarization@2.1").

    Returns:
    - result_diarize (dict): Updated metadata after assigning speaker
        labels to segments.

    Notes:
    - This function utilizes a speaker diarization model to label speaker
        segments in the audio.
    - It assigns speakers to word-level segments based on diarization results.
    - Cleans up memory by releasing resources after diarization.
    - If only one speaker is specified, each segment is automatically assigned
        as the first speaker, eliminating the need for diarization inference.
    """

    if max(min_speakers, max_speakers) > 1 and model_name:
        try:

            diarize_model = whisperx.DiarizationPipeline(
                model_name=model_name,
                use_auth_token=YOUR_HF_TOKEN,
                device=os.environ.get("SONITR_DEVICE"),
            )

        except Exception as error:
            error_str = str(error)
            gc.collect()
            torch.cuda.empty_cache()  # noqa
            if "'NoneType' object has no attribute 'to'" in error_str:
                if model_name == diarization_models["pyannote_2.1"]:
                    raise ValueError(
                        "Accept the license agreement for using Pyannote 2.1."
                        " You need to have an account on Hugging Face and "
                        "accept the license to use the models: "
                        "https://huggingface.co/pyannote/speaker-diarization "
                        "and https://huggingface.co/pyannote/segmentation "
                        "Get your KEY TOKEN here: "
                        "https://hf.co/settings/tokens "
                    )
                elif model_name == diarization_models["pyannote_3.1"]:
                    raise ValueError(
                        "New Licence Pyannote 3.1: You need to have an account"
                        " on Hugging Face and accept the license to use the "
                        "models: https://huggingface.co/pyannote/speaker-diarization-3.1 " # noqa
                        "and https://huggingface.co/pyannote/segmentation-3.0 "
                    )
            else:
                raise error
        diarize_segments = diarize_model(
            audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
        )

        result_diarize = whisperx.assign_word_speakers(
            diarize_segments, result
        )

        for segment in result_diarize["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"
                logger.warning(
                    f"No speaker detected in {segment['start']}. First TTS "
                    f"will be used for the segment text: {segment['text']} "
                )

        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()  # noqa
    else:
        result_diarize = result
        result_diarize["segments"] = [
            {**item, "speaker": "SPEAKER_00"}
            for item in result_diarize["segments"]
        ]
    return reencode_speakers(result_diarize)
