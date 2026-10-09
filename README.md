# Video Understanding Pipeline: Speech, Frame Captions and LLM Summary

A Colab notebook that summarises a video by transcribing its audio with Whisper, captioning one frame per second with a ViT-GPT2 captioner, and asking Phi-3-mini to write a summary from both, with the three model stages loaded in parallel threads.

**Case study:** [Read it](https://laiba-khawar-portfolio.vercel.app/work/video-understanding-pipeline)

## How it works

All code is in `code.ipynb`.

1. **Audio to text** (`AudioAndTextCapture`, cell 4): MoviePy extracts the audio track to `/content/audio.mp3`, then a Hugging Face `automatic-speech-recognition` pipeline with `openai/whisper-small` (30 s chunks, GPU) transcribes it.
2. **Frames to captions** (`FrameAndCaptionCapture`, cell 6): OpenCV reads the video and saves every `fps`-th frame (one per second) to `/content/frames`, then an `image-to-text` pipeline with `nlpconnect/vit-gpt2-image-captioning` captions every saved frame.
3. **LLM loading** (`LlmPipeline`, cell 7): loads `microsoft/Phi-3-mini-128k-instruct` with `device_map="cuda"` and wraps it in a `text-generation` pipeline.
4. **Parallel execution** (cell 8): the three functions run in separate `threading.Thread`s and are joined. Results are passed through global variables.
5. **Summary** (cells 9 and 10): all frame captions and the transcript are inserted into one prompt asking for a precise summary, generated greedily (`do_sample=False`, up to 250 new tokens).
6. **Cleanup** (cell 11): deletes the frames folder and the audio file.

## Results

Output is qualitative only: the notebook prints one generated summary for the test video. No accuracy, latency or quality metric is measured.

## Repository contents

- `code.ipynb`: the pipeline.
- `requirements.txt`: jupyter, moviepy, opencv-python, torch, transformers.

## Running it

Designed for Google Colab with a GPU (the recorded run used Python 3.10). All three models are placed on CUDA (`device=0` / `device_map="cuda"`), so a CUDA GPU is required as written.

1. Upload your video to Colab as `/content/video.mp4`. The input path is hardcoded in cell 2 (`video_file`, `audio_file`, `frames_folder`); edit those three variables to use another location.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   pip install accelerate "moviepy<2"
   ```

3. Run the cells in order.

Dependencies, compared with what the notebook imports (`os`, `cv2`, `torch`, `shutil`, `threading`, `warnings`, `transformers`, `moviepy.editor`):

- All imported third-party packages are listed in `requirements.txt`.
- `accelerate` is not listed but is needed for `device_map="cuda"` (the notebook's own `pip install accelerate` line is commented out).
- `from moviepy.editor import VideoFileClip` only works with MoviePy 1.x; MoviePy 2.0 removed the `moviepy.editor` module, so pin `moviepy<2`.
- The Whisper pipeline decodes the audio with `ffmpeg`, which must be installed on the system (it is preinstalled on Colab).
- `flash-attn` is optional (its install lines are commented out); without it Phi-3 prints a warning and uses standard attention.

## Known limitations

- **Hardcoded paths** under `/content/` and a single input video.
- **GPU only** as written.
- **Captions are not in time order.** Frames are listed with `os.listdir`, which does not sort `frame0.jpg`, `frame1.jpg`, ... numerically, so the captions passed to the LLM can be shuffled.
- **No timestamps.** The transcript and captions are merged into one prompt without alignment between what is said and what is shown.
- **Long videos.** All captions and the full transcript go into a single prompt with no chunking.
- **Fragile threading.** Results are shared through global variables with no error handling, so an exception in one thread only shows up later, when cell 9 finds `caption`, `text` or `LlmPipe` still set to `None`.
- Qualitative output only; nothing is evaluated.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laiba-k-00b2b1249/)
