# Configuration settings for Shannon's Guessing Game experiment

# Minimum number of completed experiments required for giveaway eligibility
GIVEAWAY_THRESHOLD = 2

# Target number of total completed experiments across all users
TOTAL_TARGET = 500

# Minimum sentence length for dataset ingestion (in characters)
# Based on Shannon's original methodology (100 chars) and replication study (avg 151 chars)
MIN_SENTENCE_LENGTH = 120

# Maximum sentence length for dataset ingestion (in characters)
MAX_SENTENCE_LENGTH = 200

# Number of characters revealed at the start of each experiment
# Helps reduce early abandonment by giving participants context
INITIAL_REVEAL_COUNT = 70

# Valid character set for the experiment
# Ukrainian alphabet (33 letters: А-Я + Є, І, Ї, Ґ) and space
# All uppercase as per Shannon's methodology
VALID_CHARACTERS = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ '

# UI cooldown timings (in milliseconds)
# Cooldown after a correct guess
COOLDOWN_MS_CORRECT = 2500
# Cooldown after an incorrect guess
COOLDOWN_MS_INCORRECT = 2500

# Number of failed guesses before showing the on-screen keyboard
KEYBOARD_REVEAL_THRESHOLD = 0