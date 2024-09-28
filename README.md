# Music-to-Text Converter

## Project Overview
The Music-to-Text Converter is a web application designed to convert audio files, specifically music, into text transcriptions. Built using **Streamlit** and leveraging the **Wav2Vec2** model from **Hugging Face**, this application aims to facilitate the transcription of spoken words within music tracks into readable text format. This can be beneficial for various use cases, including accessibility for the hearing impaired and easier content consumption for users.

## Features
- **Audio Upload**: Users can upload music files in `.wav` or `.mp3` formats.
- **Automatic Transcription**: The application automatically transcribes the audio content into text.
- **Playback**: Users can listen to the uploaded audio file directly within the application.
- **Download Option**: Users can download the generated transcription as a text file.
- **User-Friendly Interface**: The application provides clear instructions for ease of use.

## Technology Stack
- **Python**: The programming language used for backend logic.
- **Streamlit**: Framework used for building the web application.
- **PyTorch**: Deep learning library utilized for model inference.
- **Librosa**: Library for audio processing and manipulation.
- **Transformers**: Hugging Face library providing pre-trained models.

## Requirements
To run this application, ensure you have the following installed:
- Python 3.7 or higher
- Streamlit
- PyTorch
- librosa
- Hugging Face Transformers

### Installation
You can install the required packages using pip:

```bash
pip install streamlit torch librosa transformers
