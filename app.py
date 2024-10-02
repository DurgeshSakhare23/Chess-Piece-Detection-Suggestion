import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import streamlit as st
from PIL import Image
from stockfish import Stockfish

# Initialize Stockfish
stockfish_path = r"C:\Users\durge\OneDrive\Desktop\final_ai_project\archive\stockfish\stockfish-windows-x86-64-avx2.exe"
stockfish = Stockfish(stockfish_path)

# Generate a sample FEN for validation (for debugging purposes, you can replace this with your actual FEN string)
sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Check if the FEN is valid
if not stockfish.is_fen_valid(sample_fen):
    st.error("The generated FEN is not valid.")
else:
    st.write("The FEN is valid.")

# Load the trained model
model = YOLO(r'C:\Users\durge\OneDrive\Desktop\final_ai_project\archive\chess_project\Chess_Model_22\weights\best.pt')

# Define the inference function with position extraction
def run_inference_with_positions(model, image, square_size=50):
    # Run inference
    results = model(image)
    detected_pieces = []

    # Loop through the detected objects and extract their positions
    for result in results:
        boxes = result.boxes.xywh  # Extract bounding boxes (x_center, y_center, width, height)
        classes = result.boxes.cls  # Extract class labels (e.g., piece type)

        for i in range(len(boxes)):
            piece_class = model.names[int(classes[i])]  # Get class name
            x_center, y_center, _, _ = boxes[i]
            detected_pieces.append((piece_class, x_center, y_center))

    return detected_pieces

# Function to convert detected pieces to FEN
def pieces_to_fen(detected_pieces, square_size=50):
    # Create an empty 8x8 board
    board = [['1'] * 8 for _ in range(8)]
    
    # Define piece symbols for FEN
    fen_symbols = {
        'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R', 
        'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
        'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r', 
        'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p'
    }

    # Place pieces on the board according to detected positions
    for piece, x, y in detected_pieces:
        col = int(x // square_size)  # Calculate column based on x_center
        row = int(y // square_size)  # Calculate row based on y_center
        
        # Place the piece on the corresponding square (invert the row index)
        if piece in fen_symbols:
            board[7 - row][col] = fen_symbols[piece]  # Row inverted to match FEN indexing
        else:
            print(f"Warning: Detected piece '{piece}' not found in FEN symbols.")

    # Convert board to FEN string
    fen_rows = []
    for row in board:
        empty_count = 0
        fen_row = ""
        for square in row:
            if square == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += square
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    # Join the rows and construct the full FEN string
    fen_position = "/".join(fen_rows)
    return fen_position

# Function to get the best move from Stockfish
def get_best_move(detected_pieces, stockfish, turn='w', square_size=50):
    # If no pieces are detected, return "No valid moves"
    if not detected_pieces:
        return "No valid moves - No pieces detected on the board."
    
    # Check for the presence of both kings
    white_king_present = any(piece[0] == 'white-king' for piece in detected_pieces)
    black_king_present = any(piece[0] == 'black-king' for piece in detected_pieces)

    if not white_king_present or not black_king_present:
        return "Invalid state - Both kings must be present on the board."

    # Check if there is at least one playable piece (not counting kings)
    playable_pieces_count = sum(1 for piece in detected_pieces if piece[0] not in ['white-king', 'black-king'])
    
    if playable_pieces_count == 0:
        return "Invalid state - No playable pieces detected other than kings."

    # Convert detected pieces into a FEN string
    fen_position = pieces_to_fen(detected_pieces, square_size)

    # Add the current turn ('w' for white, 'b' for black) to the FEN string
    fen = fen_position + f" {turn} - - 0 1"  # FEN for current player's turn
    
    # Debugging: Print the FEN string
    st.write(f"Generated FEN: {fen}")
    
    # Set the FEN position in Stockfish
    try:
        stockfish.set_fen_position(fen)
        
        # Get the best move from Stockfish
        best_move = stockfish.get_best_move()
        
        return best_move
    except Exception as e:
        st.error(f"Error getting the best move: {e}")
        return None

# Function to draw the chessboard and pieces
def draw_chessboard_and_pieces(detected_pieces):
    # Create a chessboard
    board_size = 400  # Size of the chessboard
    square_size = board_size // 8
    chessboard = np.zeros((board_size + 100, board_size + 100, 3), dtype=np.uint8)

    # Define colors for the squares
    dark_brown = (92, 64, 51)  # Dark brown color
    brown = (165, 105, 79)     # Brown color

    # Draw the squares
    for row in range(8):
        for col in range(8):
            color = brown if (row + col) % 2 == 0 else dark_brown
            cv2.rectangle(chessboard, (col * square_size + 50, row * square_size + 50), 
                          ((col + 1) * square_size + 50, (row + 1) * square_size + 50), color, -1)

    # Overlay detected pieces
    for piece, x, y in detected_pieces:
        # Convert the center coordinates to square indices
        col = int(x // square_size)
        row = int(y // square_size)

        # Load the piece image
        piece_image_path = rf"C:\Users\durge\OneDrive\Desktop\final_ai_project\archive\piece_images\{piece}.png"
        piece_image = cv2.imread(piece_image_path, cv2.IMREAD_UNCHANGED)

        # Check if the image has an alpha channel
        if piece_image is None:
            st.warning(f"Could not find image for piece '{piece}' at {piece_image_path}")
            continue

        if piece_image.shape[2] == 4:
            b, g, r, a = cv2.split(piece_image)
            piece_image = cv2.merge((b, g, r))  # Discard alpha channel
        piece_image = cv2.resize(piece_image, (square_size, square_size))

        # Calculate the position to overlay the piece image
        y_start = row * square_size + 50
        y_end = y_start + square_size
        x_start = col * square_size + 50
        x_end = x_start + square_size
        
        # Overlay the piece image on the chessboard if sizes match
        if y_end > y_start and x_end > x_start and chessboard[y_start:y_end, x_start:x_end].shape == piece_image.shape:
            chessboard[y_start:y_end, x_start:x_end] = piece_image
        else:
            st.warning(f"Failed to overlay piece at row {row}, col {col}. Size mismatch.")

    # Add file labels (a-h) on the right side from top to bottom
    for i in range(8):
        cv2.putText(chessboard, str(i+1), (board_size + 60, 70 + i * square_size + square_size // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add rank labels (1-8) at the bottom from right to left
    for i in range(8):
        cv2.putText(chessboard, chr(97 + i), (i * square_size + 70, board_size + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the chessboard
    st.image(chessboard, channels="BGR", caption="Virtual Chessboard with Detected Pieces")

# Streamlit interface
st.title('Chess Piece Detection, Positioning, and Best Move Suggestion')

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert the uploaded image to a format compatible with OpenCV
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Run inference and get detected pieces
    detected_pieces = run_inference_with_positions(model, image)

    if detected_pieces:
        st.write(f"Detected pieces: {detected_pieces}")
    else:
        st.error("No pieces were detected. Please upload a valid image.")

    # Ask for the player's turn (w or b)
    turn = st.radio("Whose turn is it?", ('White', 'Black')).lower()[0]

    # Draw the chessboard and overlay the piece images
    draw_chessboard_and_pieces(detected_pieces)

    # Get the best move from Stockfish
    if st.button("Get Best Move"):
        best_move = get_best_move(detected_pieces, stockfish, turn)
        if best_move:
            st.success(f"Best move: {best_move}")
        else:
            st.error("No valid move could be found.")
