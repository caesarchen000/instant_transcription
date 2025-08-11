# Enhanced Instant Audio Translator

A real-time audio transcription and translation application with floating subtitle support, inspired by the [GPT-4o Transcribe Translation](https://github.com/TomokotoKiyoshi/GPT-4o-Transcribe-Translation) repository.

## ✨ New Features

### 🖥️ Floating Subtitles
- **Always-on-top overlay window** showing the latest transcription and translation
- **Draggable and resizable** - can be positioned anywhere on screen
- **Perfect for presentations** - keep subtitles visible while using other applications
- **Double-click to close** or use the toggle checkbox in the main window

### 🎯 Improved Translation Quality
- **Context-aware translation** using previous transcriptions for better continuity
- **Keyword context** - add relevant keywords to improve translation accuracy
- **Audio overlap processing** - 20% overlap between audio chunks for seamless transcription
- **Translation caching** - instant results for repeated phrases

### 🎵 Enhanced Audio Processing
- **4-second audio chunks** with 0.8-second overlap for better context
- **Real-time processing** with minimal latency
- **Better audio device detection** and handling

## 🚀 How to Use

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the application**: `python instant_translator.py`
3. **Configure languages**: Select input and output languages
4. **Add context keywords**: Enter relevant keywords for better translation
5. **Enable floating subtitles**: Check the "🖥️ Floating Subtitles" checkbox
6. **Start recording**: Click "Start Recording" and speak

## 🎮 Floating Subtitle Controls

- **Move**: Click and drag the subtitle window
- **Close**: Double-click the subtitle window or uncheck the toggle
- **Always on top**: Window stays visible over other applications
- **Real-time updates**: Shows latest transcription and translation

## 🔧 Technical Improvements

- **Context window management** for better translation continuity
- **Improved audio buffering** with optimal overlap
- **Better error handling** and user feedback
- **Responsive UI** with real-time status updates

## 📋 Requirements

- Python 3.7+
- Microphone access
- Internet connection for translation services
- See `requirements.txt` for Python packages

## 🎯 Use Cases

- **Language learning** - Real-time translation of conversations
- **Meeting documentation** - Transcribe and translate meetings
- **Presentation assistance** - Floating subtitles during presentations
- **Accessibility** - Real-time captioning for audio content
- **Content creation** - Transcribe and translate audio/video content

## 🔄 Audio Processing Flow

1. **Audio Capture**: Continuous recording in 4-second chunks
2. **Overlap Processing**: 20% overlap between chunks for continuity
3. **Transcription**: Whisper AI processes audio to text
4. **Context Enhancement**: Previous transcriptions provide context
5. **Translation**: Google Translate with context-aware prompts
6. **Real-time Display**: Updates in main window and floating subtitles

## 🎨 UI Features

- **Modern interface** with progress indicators
- **Real-time status updates** for all operations
- **Context keyword input** for better translation accuracy
- **Language selection** for multiple input/output combinations
- **Recording timer** and progress bar
- **Status indicators** for floating subtitle state

---

*Enhanced version inspired by the excellent work in the [GPT-4o Transcribe Translation](https://github.com/TomokotoKiyoshi/GPT-4o-Transcribe-Translation) repository, adapted for local processing without external API dependencies.*
