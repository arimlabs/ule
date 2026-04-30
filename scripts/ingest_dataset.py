"""
Dataset ingestion script for Shannon's Guessing Game experiment.
Loads datasets from HuggingFace, preprocesses text, and populates the database.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
import asyncio

load_dotenv()

# Add parent directory to path to import conf
sys.path.insert(0, str(Path(__file__).parent.parent))
from conf import MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH
from app.models.sentence import Sentence
from app.models.experiment_results import DatasetType
from app.db.init_db import engine, async_session, Base


def load_and_display_dataset():
    """Load the TEMP-news dataset and display its structure."""
    print("Loading dataset: a-l-o/shortnews")

    try:
        dataset = load_dataset("a-l-o/shortnews")

        print("\n" + "="*80)
        print("Dataset Structure:")
        print("="*80)
        print(dataset)

        print("\n" + "="*80)
        print("Available splits:")
        print("="*80)
        for split_name in dataset.keys():
            print(f"  - {split_name}: {len(dataset[split_name])} examples")

        print("\n" + "="*80)
        print("Column names:")
        print("="*80)
        split_name = list(dataset.keys())[0]
        print(f"  {dataset[split_name].column_names}")

        print("\n" + "="*80)
        print("First 3 examples:")
        print("="*80)
        for i, example in enumerate(dataset[split_name].select(range(min(3, len(dataset[split_name]))))):
            print(f"\nExample {i+1}:")
            for key, value in example.items():
                if isinstance(value, str):
                    print(f"  {key} (length: {len(value)} characters):")
                    print(f"    {value}")
                else:
                    print(f"  {key}: {value}")
            print("-" * 80)

        # Analyze unique characters in all text
        print("\n" + "="*80)
        print("Character Analysis:")
        print("="*80)

        all_text = ""
        for example in dataset[split_name]:
            all_text += example["text"]

        unique_chars = sorted(set(all_text))
        print(f"\nTotal unique characters: {len(unique_chars)}")
        print("\nUnique characters:")
        for char in unique_chars:
            if char == '\n':
                print(f"  '\\n' (newline)")
            elif char == ' ':
                print(f"  ' ' (space)")
            elif char == '\t':
                print(f"  '\\t' (tab)")
            else:
                print(f"  '{char}'")

        # Count character frequencies
        print("\n" + "="*80)
        print("Character Frequencies (top 20):")
        print("="*80)
        from collections import Counter
        char_counts = Counter(all_text)
        for char, count in char_counts.most_common(20):
            if char == '\n':
                char_display = '\\n'
            elif char == ' ':
                char_display = 'SPACE'
            elif char == '\t':
                char_display = '\\t'
            else:
                char_display = char
            percentage = (count / len(all_text)) * 100
            print(f"  '{char_display}': {count:5d} ({percentage:5.2f}%)")

        # Analyze core vs non-core characters (Shannon-style preprocessing)
        print("\n" + "="*80)
        print("Core vs Non-Core Character Analysis:")
        print("="*80)
        print("(Following Shannon's methodology: uppercase letters + space only)")

        # Ukrainian alphabet (33 letters including є, і, ї, ґ)
        ukrainian_core = set('АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя ')

        core_count = sum(count for char, count in char_counts.items() if char in ukrainian_core)
        non_core_count = len(all_text) - core_count

        print(f"\nCore characters (letters + space):  {core_count:6d} ({core_count/len(all_text)*100:5.2f}%)")
        print(f"Non-core (punctuation, digits, etc): {non_core_count:6d} ({non_core_count/len(all_text)*100:5.2f}%)")

        # Show what would be removed
        print("\nNon-core characters breakdown:")
        non_core_chars = {char: count for char, count in char_counts.items() if char not in ukrainian_core}
        for char, count in sorted(non_core_chars.items(), key=lambda x: -x[1])[:15]:
            if char == '\n':
                char_display = '\\n'
            elif char == '\t':
                char_display = '\\t'
            else:
                char_display = char
            percentage = (count / len(all_text)) * 100
            print(f"  '{char_display}': {count:5d} ({percentage:5.2f}%)")

        # Sentence-level analysis: filter vs preprocess
        print("\n" + "="*80)
        print("Sentence-Level Analysis:")
        print("="*80)
        print("Comparing: filtering out sentences with non-core chars vs preprocessing all")

        import re
        # Split into sentences (roughly - by . ! ?)
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        print(f"\nTotal sentences: {len(sentences)}")

        # Count sentences with non-core characters
        sentences_with_noncore = 0
        sentences_pure = 0
        for sentence in sentences:
            has_noncore = any(char not in ukrainian_core for char in sentence)
            if has_noncore:
                sentences_with_noncore += 1
            else:
                sentences_pure += 1

        print(f"Sentences with ONLY core characters: {sentences_pure} ({sentences_pure/len(sentences)*100:.2f}%)")
        print(f"Sentences with non-core characters:  {sentences_with_noncore} ({sentences_with_noncore/len(sentences)*100:.2f}%)")

        # Show examples of sentences with non-core chars
        print("\n" + "="*80)
        print("Sample sentences with non-core characters:")
        print("="*80)

        # Calculate non-core percentage for each sentence
        sentence_noncore_stats = []
        for sentence in sentences:
            noncore_count = sum(1 for char in sentence if char not in ukrainian_core)
            noncore_percentage = (noncore_count / len(sentence) * 100) if len(sentence) > 0 else 0
            sentence_noncore_stats.append((sentence, noncore_count, noncore_percentage))

        # Sort by non-core percentage
        sentence_noncore_stats.sort(key=lambda x: x[2], reverse=True)

        print("\nTop 5 sentences with highest non-core character percentage:")
        for i, (sentence, noncore_count, noncore_pct) in enumerate(sentence_noncore_stats[:5]):
            noncore_chars = set(char for char in sentence if char not in ukrainian_core)
            print(f"\n{i+1}. Non-core: {noncore_pct:.2f}% ({noncore_count}/{len(sentence)} chars)")
            print(f"   Original: {sentence}")
            # Show what it would look like after preprocessing
            preprocessed = ''.join(char.upper() if char in ukrainian_core else '' for char in sentence)
            preprocessed = ' '.join(preprocessed.split())  # Normalize spaces
            print(f"   After preprocessing: {preprocessed}")
            print(f"   Non-core chars: {sorted(noncore_chars)}")

        # Analyze clean sentences (letters + punctuation only, no digits/Latin)
        print("\n" + "="*80)
        print("Clean Sentence Analysis (no digits or Latin letters):")
        print("="*80)

        clean_sentences = []
        for sentence in sentences:
            # Check if sentence has digits or Latin letters
            has_digits = any(char.isdigit() for char in sentence)
            has_latin = any(char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' for char in sentence)

            if not has_digits and not has_latin:
                clean_sentences.append(sentence)

        print(f"\nSentences WITHOUT digits or Latin letters: {len(clean_sentences)} ({len(clean_sentences)/len(sentences)*100:.2f}%)")
        print(f"Sentences WITH digits or Latin letters:    {len(sentences) - len(clean_sentences)} ({(len(sentences) - len(clean_sentences))/len(sentences)*100:.2f}%)")

        if len(clean_sentences) > 0:
            print("\nSample clean sentences (first 10):")
            for i, sentence in enumerate(clean_sentences[:10]):
                # Calculate length after preprocessing
                preprocessed = ''.join(char.upper() if char in ukrainian_core else '' for char in sentence)
                preprocessed = ' '.join(preprocessed.split())
                noncore_chars = set(char for char in sentence if char not in ukrainian_core)
                print(f"\n{i+1}. Length: {len(preprocessed)} chars after preprocessing")
                print(f"   Original: {sentence}")
                print(f"   Preprocessed: {preprocessed}")
                if noncore_chars:
                    print(f"   Only punctuation removed: {sorted(noncore_chars)}")

            # Statistics on clean sentences
            clean_lengths = []
            for sentence in clean_sentences:
                preprocessed = ''.join(char.upper() if char in ukrainian_core else '' for char in sentence)
                preprocessed = ' '.join(preprocessed.split())
                clean_lengths.append(len(preprocessed))

            if clean_lengths:
                import statistics
                print(f"\nClean sentence length statistics (after preprocessing):")
                print(f"  Mean:   {statistics.mean(clean_lengths):.1f} characters")
                print(f"  Median: {statistics.median(clean_lengths):.1f} characters")
                print(f"  Min:    {min(clean_lengths)} characters")
                print(f"  Max:    {max(clean_lengths)} characters")

        # Analyze each punctuation character with examples
        print("\n" + "="*80)
        print("Punctuation Character Examples:")
        print("="*80)

        # Collect all unique punctuation from clean sentences
        all_punctuation = set()
        for sentence in clean_sentences:
            for char in sentence:
                if char not in ukrainian_core:
                    all_punctuation.add(char)

        print(f"\nAll punctuation marks found in clean sentences: {sorted(all_punctuation)}")

        # For each punctuation, show examples
        for punct in sorted(all_punctuation):
            if punct == '\n':
                continue
            print(f"\n--- Character: '{punct}' ---")
            examples_shown = 0
            for sentence in clean_sentences:
                if punct in sentence:
                    # Find context around the punctuation
                    idx = sentence.index(punct)
                    start = max(0, idx - 40)
                    end = min(len(sentence), idx + 40)
                    context = sentence[start:end]

                    # Highlight the punctuation
                    highlight_idx = idx - start
                    before = context[:highlight_idx]
                    after = context[highlight_idx+1:]

                    print(f"  Example {examples_shown + 1}: ...{before}[{punct}]{after}...")

                    # Show how it looks after preprocessing
                    preprocessed_context = ''.join(c.upper() if c in ukrainian_core else ' ' for c in context)
                    preprocessed_context = ' '.join(preprocessed_context.split())
                    print(f"              After: ...{preprocessed_context}...")

                    examples_shown += 1
                    if examples_shown >= 3:
                        break

        # Final preprocessing and output
        print("\n" + "="*80)
        print("FINAL PREPROCESSED SENTENCES:")
        print("="*80)
        print(f"\nProcessing {len(clean_sentences)} clean sentences...")

        def preprocess_sentence(text):
            """
            Preprocess text following Shannon's methodology:
            - Convert to uppercase
            - Keep only Ukrainian letters (А-Я + Є, І, Ї, Ґ) and spaces
            - Normalize multiple spaces to single space
            """
            # Define Ukrainian alphabet
            ukrainian_letters = set('АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя ')

            # Convert to uppercase and filter
            result = ''.join(char.upper() if char in ukrainian_letters else ' ' for char in text)

            # Normalize spaces
            result = ' '.join(result.split())

            return result

        preprocessed_sentences = []
        for sentence in clean_sentences:
            preprocessed = preprocess_sentence(sentence)
            if preprocessed:  # Only add non-empty sentences
                preprocessed_sentences.append(preprocessed)

        print(f"\nSuccessfully preprocessed: {len(preprocessed_sentences)} sentences")

        # Apply length filter (both minimum and maximum)
        print("\n" + "="*80)
        print("LENGTH FILTERING:")
        print("="*80)
        print(f"\nSentence length range: {MIN_SENTENCE_LENGTH}-{MAX_SENTENCE_LENGTH} characters")
        print("(Based on Shannon's original methodology and replication study)")

        # Show distribution before filtering
        lengths_before = [len(s) for s in preprocessed_sentences]
        import statistics

        # Filter by minimum and maximum length
        filtered_sentences = [s for s in preprocessed_sentences if MIN_SENTENCE_LENGTH <= len(s) <= MAX_SENTENCE_LENGTH]
        lengths_after = [len(s) for s in filtered_sentences]

        print(f"\nBefore filtering:")
        print(f"  Total: {len(preprocessed_sentences)} sentences")
        print(f"  Mean:   {statistics.mean(lengths_before):.1f} characters")
        print(f"  Median: {statistics.median(lengths_before):.1f} characters")
        print(f"  Min:    {min(lengths_before)} characters")
        print(f"  Max:    {max(lengths_before)} characters")

        print(f"\nAfter filtering ({MIN_SENTENCE_LENGTH}-{MAX_SENTENCE_LENGTH} chars):")
        print(f"  Total: {len(filtered_sentences)} sentences")
        if filtered_sentences:
            print(f"  Mean:   {statistics.mean(lengths_after):.1f} characters")
            print(f"  Median: {statistics.median(lengths_after):.1f} characters")
            print(f"  Min:    {min(lengths_after)} characters")
            print(f"  Max:    {max(lengths_after)} characters")
            print(f"  StdDev: {statistics.stdev(lengths_after):.1f} characters")
        else:
            print("  No sentences meet the length requirements!")

        print(f"\nRemoved: {len(preprocessed_sentences) - len(filtered_sentences)} sentences ({(len(preprocessed_sentences) - len(filtered_sentences))/len(preprocessed_sentences)*100:.1f}%)")
        print(f"Retained: {len(filtered_sentences)} sentences ({len(filtered_sentences)/len(preprocessed_sentences)*100:.1f}%)")

        print(f"\nAll filtered sentences:\n")
        for i, sentence in enumerate(filtered_sentences, 1):
            print(f"{i:3d}. [{len(sentence):3d} chars] {sentence}")

        # Summary statistics
        print("\n" + "="*80)
        print("FINAL SUMMARY:")
        print("="*80)
        print(f"\nTotal sentences ready for database: {len(filtered_sentences)}")
        print(f"Character set: Ukrainian alphabet (А-Я + Є, І, Ї, Ґ) + space = 34 characters")
        print(f"Length range: {MIN_SENTENCE_LENGTH}-{MAX_SENTENCE_LENGTH} characters")
        if filtered_sentences:
            print(f"\nLength statistics:")
            print(f"  Mean:   {statistics.mean(lengths_after):.1f} characters")
            print(f"  Median: {statistics.median(lengths_after):.1f} characters")
            print(f"  Min:    {min(lengths_after)} characters")
            print(f"  Max:    {max(lengths_after)} characters")
            print(f"  StdDev: {statistics.stdev(lengths_after):.1f} characters")

        return filtered_sentences

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


async def insert_sentences_to_db(sentences, dataset_type=DatasetType.FORMAL, source="a-l-o/TEMP-news"):
    """Insert preprocessed sentences into the database."""

    print("\n" + "="*80)
    print("DATABASE INSERTION:")
    print("="*80)
    print(f"\nConnecting to database...")

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        try:
            # Check if sentences already exist
            print(f"Checking for existing sentences...")
            from sqlalchemy import select, func as sql_func
            result = await session.execute(select(sql_func.count(Sentence.id)))
            existing_count = result.scalar()

            if existing_count > 0:
                print(f"\nWarning: Database already contains {existing_count} sentences.")
                response = input("Do you want to continue and add more sentences? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Insertion cancelled.")
                    return

            # Insert sentences
            print(f"\nInserting {len(sentences)} sentences...")
            inserted = 0

            for sentence_text in sentences:
                sentence = Sentence(
                    text=sentence_text,
                    dataset_type=dataset_type,
                    source=source,
                    length=len(sentence_text)
                )
                session.add(sentence)
                inserted += 1

                if inserted % 10 == 0:
                    print(f"  Inserted {inserted}/{len(sentences)} sentences...")

            await session.commit()

            print(f"\n✓ Successfully inserted {inserted} sentences into the database!")

            # Verify insertion
            result = await session.execute(select(sql_func.count(Sentence.id)))
            total_count = result.scalar()
            print(f"✓ Total sentences in database: {total_count}")

        except Exception as e:
            await session.rollback()
            print(f"\n✗ Error inserting sentences: {e}")
            raise


async def main():
    """Main function to load dataset and insert into database."""
    print("="*80)
    print("SHANNON'S GUESSING GAME - DATASET INGESTION")
    print("="*80)

    # Load and preprocess dataset
    filtered_sentences = load_and_display_dataset()

    if not filtered_sentences:
        print("\n✗ No sentences to insert!")
        return

    # Ask user if they want to insert into database
    print("\n" + "="*80)
    response = input(f"\nInsert {len(filtered_sentences)} sentences into the database? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        await insert_sentences_to_db(
            filtered_sentences,
            dataset_type=DatasetType.FORMAL,
            source="a-l-o/TEMP-news"
        )
    else:
        print("Insertion skipped.")


if __name__ == "__main__":
    asyncio.run(main())