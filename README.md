# Chess-Piece-Detection-Suggestion

This project implements a chess piece detection system using the YOLOv8 model and a best move suggestion system using the Stockfish chess engine. The goal is to automatically detect the positions of the chess pieces from an image, convert the detected pieces into a valid FEN string, and use Stockfish to suggest the next best move.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [YOLOv8 for Chess Piece Detection](#yolov8-for-chess-piece-detection)
- [Stockfish for Best Move Suggestion](#stockfish-for-best-move-suggestion)
- [How to Run the Project](#how-to-run-the-project)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project involves two primary tasks:
1. Detecting chess pieces from an image using the YOLOv8 object detection model.
2. Suggesting the best next move using the Stockfish chess engine after analyzing the board state.

## Technologies Used

- **Python**: The main programming language used.
- **YOLOv8**: A state-of-the-art object detection model used to detect chess pieces.
- **Stockfish**: A powerful open-source chess engine used for move suggestion.
- **Streamlit**: A web-based framework used for creating the user interface.
- **OpenCV**: For image processing and visualization.
- **PIL (Pillow)**: For handling image input/output.
- **Numpy**: For handling matrix and array operations.

## Setup Instructions

### Prerequisites

1. **Python 3.8+** should be installed.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt

# Dependencies
1. ultralytics (YOLOv8)
2. stockfish
3. opencv-python
4. streamlit
5. pillow
6. matplotlib
7. numpy

# Model and Engine Setup
YOLOv8 Model: Download or train your YOLOv8 model and ensure the model weights are available in the project directory.
Stockfish: Download and install the Stockfish chess engine. Ensure the correct path to the Stockfish executable is set in the script.

How It Works ? 
# YOLOv8 for Chess Piece Detection
The YOLOv8 model is used to detect chess pieces on a board. The model is trained to recognize different chess pieces such as white/black pawns, knights, bishops, rooks, queens, and kings. The model outputs the position and class of each detected piece.

# Stockfish for Best Move Suggestion
Once the chess pieces are detected and their positions are converted into a FEN string, Stockfish takes over to analyze the board state and suggest the best next move based on the current position. Stockfish uses advanced chess algorithms to compute the best possible move.

# Process Overview
An image of the chessboard is uploaded through the user interface.
The YOLOv8 model detects the positions and types of chess pieces on the board.
Detected pieces are converted into a FEN string.
The FEN string is fed into the Stockfish engine, which suggests the best next move.
The results are displayed, including the best move and a visualization of the chessboard.

# How to Run the Project
# Clone the repository:
git clone https://github.com/yourusername/chess-detection-best-move.git
cd chess-detection-best-move

# Install the required dependencies:
pip install -r requirements.txt

# Run the project using Streamlit:
streamlit run app.py


### Explanation:

- **Project Overview**: Describes the main objectives of the project.
- **Technologies Used**: Lists the primary technologies and libraries used in the project.
- **Setup Instructions**: Gives step-by-step instructions on how to set up and run the project.
- **How It Works**: Explains the core functionality of chess piece detection and the best move suggestion.
- **How to Run the Project**: Guides users on how to clone the repository and execute the application.
- **Future Improvements**: Suggests potential areas to enhance the project.
- **Acknowledgments**: Credits relevant libraries and frameworks.


