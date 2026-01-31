import re
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='FSKU Subjective Benchmark Evaluation')
    parser.add_argument(
        '--prediction',
        type=str,
        default='./subjective_submission.csv',
        help='Path to prediction CSV file (default: ./subjective_submission.csv)'
    )
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Path to test CSV file with ground truth (optional, auto-detected by lang)'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='kor',
        choices=['kor', 'eng'],
        help='Language for auto-detecting test file (default: kor)'
    )
    parser.add_argument(
        '--similarity_model',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        help='Sentence similarity model (default: paraphrase-multilingual-MiniLM-L12-v2)'
    )
    parser.add_argument(
        '--similarity_weight',
        type=float,
        default=0.6,
        help='Weight for similarity score (default: 0.6)'
    )
    parser.add_argument(
        '--keyword_weight',
        type=float,
        default=0.4,
        help='Weight for keyword score (default: 0.4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./subjective_evaluation_results.csv',
        help='Output file path (default: ./subjective_evaluation_results.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed evaluation results'
    )
    return parser.parse_args()


def load_similarity_model(model_name: str):
    """
    Load sentence transformer model for similarity calculation.

    Args:
        model_name: Name or path of the sentence transformer model

    Returns:
        SentenceTransformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading similarity model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )


def calculate_similarity(model, text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts.

    Args:
        model: SentenceTransformer model
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score (0-1)
    """
    from sentence_transformers import util

    # Handle empty strings
    if not text1 or not text2:
        return 0.0

    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity.cpu().numpy())


def parse_keywords(keyword_str: str) -> List[str]:
    """
    Parse keywords from comma-separated string.

    Args:
        keyword_str: Comma-separated keywords string

    Returns:
        List of keywords (lowercase, stripped)
    """
    if pd.isna(keyword_str) or not keyword_str:
        return []

    # Split by comma and clean
    keywords = [k.strip().lower() for k in str(keyword_str).split(',')]
    return [k for k in keywords if k]


def calculate_keyword_score(predicted: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Calculate keyword matching score.

    Args:
        predicted: Predicted answer text
        keywords: List of required keywords

    Returns:
        Tuple of (score, matched_keywords, missing_keywords)
    """
    if not keywords:
        return 1.0, [], []

    predicted_lower = predicted.lower()
    matched = []
    missing = []

    for keyword in keywords:
        # Check if keyword or its parts are in the predicted text
        keyword_parts = keyword.split()
        if keyword in predicted_lower:
            matched.append(keyword)
        elif len(keyword_parts) > 1 and all(part in predicted_lower for part in keyword_parts):
            matched.append(keyword)
        else:
            missing.append(keyword)

    score = len(matched) / len(keywords) if keywords else 1.0
    return score, matched, missing


def evaluate_single(
    model,
    predicted: str,
    reference: str,
    keywords: List[str],
    similarity_weight: float,
    keyword_weight: float
) -> dict:
    """
    Evaluate a single prediction.

    Args:
        model: SentenceTransformer model
        predicted: Predicted answer
        reference: Reference answer
        keywords: Required keywords
        similarity_weight: Weight for similarity score
        keyword_weight: Weight for keyword score

    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate similarity score
    similarity_score = calculate_similarity(model, predicted, reference)

    # Calculate keyword score
    keyword_score, matched_keywords, missing_keywords = calculate_keyword_score(predicted, keywords)

    # Calculate combined score
    combined_score = (similarity_weight * similarity_score) + (keyword_weight * keyword_score)

    return {
        'similarity_score': similarity_score,
        'keyword_score': keyword_score,
        'combined_score': combined_score,
        'matched_keywords': matched_keywords,
        'missing_keywords': missing_keywords,
        'num_keywords': len(keywords),
        'num_matched': len(matched_keywords)
    }


def print_evaluation_report(results: dict, detailed_results: List[dict], verbose: bool = False) -> None:
    """
    Print formatted evaluation report.

    Args:
        results: Dictionary containing overall evaluation metrics
        detailed_results: List of per-question evaluation results
        verbose: If True, print detailed per-question results
    """
    print("\n" + "=" * 70)
    print("FSKU Subjective Benchmark Evaluation Report")
    print("=" * 70)
    print(f"Total Questions: {results['total']}")
    print(f"\nScore Weights: Similarity={results['similarity_weight']:.0%}, Keyword={results['keyword_weight']:.0%}")
    print("-" * 70)
    print(f"Average Similarity Score: {results['avg_similarity']:.4f} ({results['avg_similarity']*100:.2f}%)")
    print(f"Average Keyword Score:    {results['avg_keyword']:.4f} ({results['avg_keyword']*100:.2f}%)")
    print(f"Average Combined Score:   {results['avg_combined']:.4f} ({results['avg_combined']*100:.2f}%)")
    print("-" * 70)
    print(f"Total Keywords: {results['total_keywords']}")
    print(f"Matched Keywords: {results['total_matched']} ({results['keyword_match_rate']*100:.2f}%)")
    print("=" * 70)

    # Score distribution
    print("\nScore Distribution:")
    print("-" * 70)
    scores = [r['combined_score'] for r in detailed_results]
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for low, high in bins:
        count = sum(1 for s in scores if low <= s < high or (high == 1.0 and s == 1.0))
        bar = '█' * (count * 2)
        print(f"  {low:.1f}-{high:.1f}: {bar} ({count})")

    if verbose:
        print("\n" + "=" * 70)
        print("Detailed Results (Top 5 and Bottom 5 by Combined Score)")
        print("=" * 70)

        sorted_results = sorted(enumerate(detailed_results), key=lambda x: x[1]['combined_score'], reverse=True)

        print("\n[Top 5]")
        for idx, result in sorted_results[:5]:
            print(f"\nQ{idx+1}: Combined={result['combined_score']:.4f} "
                  f"(Sim={result['similarity_score']:.4f}, Kw={result['keyword_score']:.4f})")
            if result['matched_keywords']:
                print(f"  Matched: {', '.join(result['matched_keywords'])}")

        print("\n[Bottom 5]")
        for idx, result in sorted_results[-5:]:
            print(f"\nQ{idx+1}: Combined={result['combined_score']:.4f} "
                  f"(Sim={result['similarity_score']:.4f}, Kw={result['keyword_score']:.4f})")
            if result['missing_keywords']:
                print(f"  Missing: {', '.join(result['missing_keywords'])}")


def main():
    args = parse_args()

    # Auto-detect test file if not specified
    test_path = args.test if args.test else f'./subjective_test({args.lang}).csv'

    print("=" * 70)
    print("FSKU Subjective Benchmark Evaluation")
    print("=" * 70)
    print(f"Prediction file: {args.prediction}")
    print(f"Test file: {test_path}")
    print(f"Similarity model: {args.similarity_model}")
    print(f"Weights: Similarity={args.similarity_weight}, Keyword={args.keyword_weight}")
    print("=" * 70)

    # Validate weights
    if abs(args.similarity_weight + args.keyword_weight - 1.0) > 0.001:
        print("Warning: Weights do not sum to 1.0. Normalizing...")
        total = args.similarity_weight + args.keyword_weight
        args.similarity_weight /= total
        args.keyword_weight /= total

    # Load data
    print("\nLoading data...")
    try:
        predictions_df = pd.read_csv(args.prediction)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure prediction file exists (run subjective_baseline.py first)")
        return

    # Validate data
    if len(predictions_df) != len(test_df):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions_df)}) vs test ({len(test_df)})"
        )

    print(f"Loaded {len(predictions_df)} predictions")

    # Load similarity model
    model = load_similarity_model(args.similarity_model)

    # Evaluate each prediction
    print("\nEvaluating predictions...")
    detailed_results = []

    for i in range(len(predictions_df)):
        predicted = str(predictions_df['Predicted_Answer'].iloc[i]) if 'Predicted_Answer' in predictions_df.columns else str(predictions_df['Answer'].iloc[i])
        reference = str(test_df['Answer'].iloc[i])
        keywords = parse_keywords(test_df['Keyword'].iloc[i])

        result = evaluate_single(
            model=model,
            predicted=predicted,
            reference=reference,
            keywords=keywords,
            similarity_weight=args.similarity_weight,
            keyword_weight=args.keyword_weight
        )
        detailed_results.append(result)

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(predictions_df)} questions...")

    # Calculate overall metrics
    total_keywords = sum(r['num_keywords'] for r in detailed_results)
    total_matched = sum(r['num_matched'] for r in detailed_results)

    overall_results = {
        'total': len(detailed_results),
        'avg_similarity': np.mean([r['similarity_score'] for r in detailed_results]),
        'avg_keyword': np.mean([r['keyword_score'] for r in detailed_results]),
        'avg_combined': np.mean([r['combined_score'] for r in detailed_results]),
        'total_keywords': total_keywords,
        'total_matched': total_matched,
        'keyword_match_rate': total_matched / total_keywords if total_keywords > 0 else 0,
        'similarity_weight': args.similarity_weight,
        'keyword_weight': args.keyword_weight
    }

    # Print report
    print_evaluation_report(overall_results, detailed_results, verbose=args.verbose)

    # Save detailed results
    results_df = pd.DataFrame({
        'Question': test_df['Question'],
        'Reference_Answer': test_df['Answer'],
        'Predicted_Answer': predictions_df['Predicted_Answer'] if 'Predicted_Answer' in predictions_df.columns else predictions_df['Answer'],
        'Keywords': test_df['Keyword'],
        'Similarity_Score': [r['similarity_score'] for r in detailed_results],
        'Keyword_Score': [r['keyword_score'] for r in detailed_results],
        'Combined_Score': [r['combined_score'] for r in detailed_results],
        'Matched_Keywords': [', '.join(r['matched_keywords']) for r in detailed_results],
        'Missing_Keywords': [', '.join(r['missing_keywords']) for r in detailed_results]
    })
    results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nDetailed results saved to: {args.output}")

    # Save summary
    summary_path = args.output.replace('.csv', '_summary.csv')
    summary_df = pd.DataFrame([overall_results])
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
