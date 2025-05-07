import Jetson.GPIO as GPIO
# from make_wttr_location import llm_parse_for_wttr
# from get_weather_from_wttr import get_weather
# from speech_recognition import get_whisper_transcription, init_whisper
import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
from requests import get
from ollama import Client

def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


def record_audio(duration_seconds: int = 5) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)

def init_whisper() -> Pipeline:
    model_id="distil-whisper/distil-medium.en"
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = build_pipeline(model_id, torch_dtype, device)

    return pipe

def get_whisper_transcription(pipe: Pipeline) -> str:

    audio = record_audio()

    speech = pipe(audio)

    print(speech)

    return speech

def get_weather(location: str) -> None:
    url = f"https://wttr.in/{location}"

    try:
        print(f"Attempting to get weather from {location}")
        response = get(url) 
        response.raise_for_status()
    
    except Exception as e:
        print(f"Error occured getting the weather for {location}: {e}")

    else:
        print("\n")
        print(response.text)


def llm_parse_for_wttr(raw_input: str) -> str:
    """Parse the input using the LLM and return the result."""
    
    try:
        LLM_MODEL: str = "gemma3:27b"  #  this is running on the AI server
        client: Client = Client(host="http://ai.dfec.xyz:11434")  # this is the AI server

        response = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You are a weather assistant. Your job is to parse the input text and extract the location for the wttr.in API.
                        The location should be formatted as follows:
                        - All spaces should be replaced with plus signs (+).
                        - For cities, use the city name (e.g., "New+York").
                        - For regions, use the region name (e.g., "California").
                        - For airports, use the IATA code (e.g., "LAX").
                        - For specific locations or landmarks, use a tilde (~) before the location (e.g., "~Grand+Canyon").
                        - If the city and state are given, use the city name (e.g., "Seattle, Washington" -> "Seattle")
                        """,
                },
                {"role": "user", "content": raw_input},
            ],
            model="gemma3:27b",
        )

        return response["message"]["content"]

    except Exception as e:
        print(f"Error occured parsing the location for wttr.in: {e}")
        return None

# Set up GPIO pins
GPIO.setmode(GPIO.BOARD)
input_pin = 7
GPIO.setup(input_pin, GPIO.IN) # reading input on this pin


# """
# Expecting pull-down network with button connected to `input_pin`

#     ^
#     |      _____ button
#     |     /
#     |    /
#     -- \----- input_pin
#     |
#     \
#     /
#     |
#     _
#     -
# """

def main():
    try:

        print("Initializing pipeline...")
        pipe = init_whisper()

        while True:

            print("Waiting for input...")

            # blocks function until rising edge detected
            GPIO.wait_for_edge(input_pin, GPIO.RISING) # wait for rising edge of input pin

            # run the whisper model here and return the transcription

            # get keyboard input to test next section
            print("Ready to find weather information")

            # transcription = input()
            transcription = get_whisper_transcription(pipe)

            if transcription:
                transcription = transcription["text"]

            wttr_location_query = llm_parse_for_wttr(transcription) # get the formatted location for wttr.in
            if wttr_location_query:
                get_weather(wttr_location_query)
            else:
                print("Failed to get a location")

    except Exception as e:
        print(f"Error in main(): {e}")


if __name__ == "__main__":
    main()