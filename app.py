"""Run the trained SignBridge A-Z LSTM webcam tester."""
import sys
from lstm_webcam import main

if __name__ == '__main__':
    # Preserve the existing launch command; the LSTM is now also the default.
    main([arg for arg in sys.argv[1:] if arg != '--lstm'])
