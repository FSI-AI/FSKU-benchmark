import re
import argparse
import pandas as pd
from typing import Tuple


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='FSKU Benchmark Evaluation')
    parser.add_argument(
        '--prediction',
        type=str,
        default='./baseline_submission.csv',
        help='Path to prediction CSV file (default: ./baseline_submission.csv)'
    )
    parser.add_argument(
        '--answer',
        type=str,
        default='./answer.csv',
        help='Path to answer CSV file (default: ./answer.csv)'
    )
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='Path to test CSV file for domain-level evaluation (optional)'
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='kor',
        choices=['kor', 'eng'],
        help='Language for auto-detecting test file (default: kor)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed incorrect predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./evaluation_results.csv',
        help='Output file path for detailed results (default: ./evaluation_results.csv)'
    )
    return parser.parse_args()


def load_data(prediction_path: str, answer_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load prediction and answer CSV files.

    Args:
        prediction_path: Path to the prediction CSV file (baseline_submission.csv)
        answer_path: Path to the ground truth CSV file (answer.csv)

    Returns:
        Tuple of (predictions DataFrame, answers DataFrame)
    """
    predictions = pd.read_csv(prediction_path)
    answers = pd.read_csv(answer_path)
    return predictions, answers


def extract_answer_number(answer_text: str) -> str:
    """
    Extract the answer number from answer text.

    The answer format in answer.csv is a numeric string (e.g., "2").
    This function handles both numeric-only and legacy "number + text" formats.

    Args:
        answer_text: Answer text from answer.csv

    Returns:
        Extracted answer number as string, or "0" if extraction fails
    """
    if pd.isna(answer_text):
        return "0"

    answer_str = str(answer_text).strip()

    # Extract leading number (1-99)
    match = re.match(r"^\s*([1-9][0-9]?)\s", answer_str)
    if match:
        return match.group(1)

    # Try to match just a number
    match = re.match(r"^\s*([1-9][0-9]?)$", answer_str)
    if match:
        return match.group(1)

    return "0"


def evaluate(predictions: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """
    Evaluate predictions against ground truth answers.

    Args:
        predictions: DataFrame with 'Answer' column containing predicted answers
        answers: DataFrame with 'Answer' column containing ground truth answers

    Returns:
        Dictionary containing evaluation metrics
    """
    # Extract answer numbers from ground truth
    ground_truth = answers['Answer'].apply(extract_answer_number).tolist()

    # Get predicted answers
    pred_answers = predictions['Answer'].apply(lambda x: str(x).strip()).tolist()

    # Validate lengths match
    if len(pred_answers) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions ({len(pred_answers)}) vs answers ({len(ground_truth)})"
        )

    # Calculate metrics
    total = len(ground_truth)
    correct = 0
    results = []

    for i, (pred, gt) in enumerate(zip(pred_answers, ground_truth)):
        is_correct = pred == gt
        if is_correct:
            correct += 1
        results.append({
            'index': i,
            'predicted': pred,
            'ground_truth': gt,
            'correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy,
        'accuracy_percent': f"{accuracy * 100:.2f}%",
        'detailed_results': results
    }


def evaluate_by_domain(predictions: pd.DataFrame, answers: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Evaluate predictions by domain and sub-domain.

    Args:
        predictions: DataFrame with 'Answer' column containing predicted answers
        answers: DataFrame with 'Answer' column containing ground truth answers
        test: DataFrame with 'Domain' and 'Sub_domain' columns

    Returns:
        Dictionary containing domain-level evaluation metrics
    """
    # Extract answer numbers
    ground_truth = answers['Answer'].apply(extract_answer_number).tolist()
    pred_answers = predictions['Answer'].apply(lambda x: str(x).strip()).tolist()

    # Add predictions and ground truth to test dataframe
    eval_df = test.copy()
    eval_df['predicted'] = pred_answers
    eval_df['ground_truth'] = ground_truth
    eval_df['correct'] = eval_df['predicted'] == eval_df['ground_truth']

    # Calculate domain-level metrics
    domain_results = {}
    for domain in eval_df['Domain'].unique():
        domain_df = eval_df[eval_df['Domain'] == domain]
        domain_total = len(domain_df)
        domain_correct = domain_df['correct'].sum()
        domain_accuracy = domain_correct / domain_total if domain_total > 0 else 0.0

        # Calculate sub-domain metrics
        subdomain_results = {}
        for subdomain in domain_df['Sub_domain'].unique():
            sub_df = domain_df[domain_df['Sub_domain'] == subdomain]
            sub_total = len(sub_df)
            sub_correct = sub_df['correct'].sum()
            sub_accuracy = sub_correct / sub_total if sub_total > 0 else 0.0
            subdomain_results[subdomain] = {
                'total': sub_total,
                'correct': int(sub_correct),
                'accuracy': sub_accuracy,
                'accuracy_percent': f"{sub_accuracy * 100:.2f}%"
            }

        domain_results[domain] = {
            'total': domain_total,
            'correct': int(domain_correct),
            'accuracy': domain_accuracy,
            'accuracy_percent': f"{domain_accuracy * 100:.2f}%",
            'subdomains': subdomain_results
        }

    return domain_results


def print_evaluation_report(results: dict, verbose: bool = False) -> None:
    """
    Print formatted evaluation report.

    Args:
        results: Dictionary containing evaluation metrics from evaluate()
        verbose: If True, print details of all incorrect predictions
    """
    print("=" * 60)
    print("FSKU Benchmark Evaluation Report")
    print("=" * 60)
    print(f"Total Questions: {results['total']}")
    print(f"Correct Answers: {results['correct']}")
    print(f"Incorrect Answers: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy_percent']} ({results['accuracy']:.4f})")
    print("=" * 60)

    if verbose and results['detailed_results']:
        incorrect = [r for r in results['detailed_results'] if not r['correct']]
        if incorrect:
            print(f"\nIncorrect Predictions ({len(incorrect)} items):")
            print("-" * 60)
            for item in incorrect[:20]:  # Show first 20
                print(f"  Q{item['index'] + 1}: Predicted={item['predicted']}, Ground Truth={item['ground_truth']}")
            if len(incorrect) > 20:
                print(f"  ... and {len(incorrect) - 20} more")
            print("-" * 60)


def print_domain_report(domain_results: dict) -> None:
    """
    Print domain-level evaluation report.

    Args:
        domain_results: Dictionary containing domain-level metrics
    """
    print("\n" + "=" * 60)
    print("Domain-Level Evaluation Report")
    print("=" * 60)

    for domain, metrics in domain_results.items():
        print(f"\n[{domain}]")
        print(f"  Total: {metrics['total']} | Correct: {metrics['correct']} | Accuracy: {metrics['accuracy_percent']}")
        print(f"  Sub-domains:")
        for subdomain, sub_metrics in metrics['subdomains'].items():
            print(f"    - {subdomain}: {sub_metrics['accuracy_percent']} ({sub_metrics['correct']}/{sub_metrics['total']})")

    print("\n" + "=" * 60)


def main():
    """
    Main function to run evaluation.
    """
    args = parse_args()

    print("=" * 60)
    print("FSKU Benchmark Evaluation")
    print("=" * 60)
    print(f"Prediction file: {args.prediction}")
    print(f"Answer file: {args.answer}")
    print("=" * 60)

    # Load prediction and answer data
    print("\nLoading data...")
    try:
        predictions, answers = load_data(args.prediction, args.answer)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure prediction file exists (run baseline.py first)")
        return

    print(f"Loaded {len(predictions)} predictions")

    # Run evaluation
    print("Evaluating predictions...")
    results = evaluate(predictions, answers)

    # Print overall report
    print_evaluation_report(results, verbose=args.verbose)

    # Domain-level evaluation if test file is provided
    test_path = args.test
    if test_path is None:
        # Auto-detect test file based on language
        test_path = f'./test({args.lang}).csv'

    try:
        test = pd.read_csv(test_path)
        if 'Domain' in test.columns and 'Sub_domain' in test.columns:
            print(f"\nRunning domain-level evaluation using: {test_path}")
            domain_results = evaluate_by_domain(predictions, answers, test)
            print_domain_report(domain_results)

            # Save domain results
            domain_output = args.output.replace('.csv', '_by_domain.csv')
            domain_rows = []
            for domain, metrics in domain_results.items():
                for subdomain, sub_metrics in metrics['subdomains'].items():
                    domain_rows.append({
                        'Domain': domain,
                        'Sub_domain': subdomain,
                        'Total': sub_metrics['total'],
                        'Correct': sub_metrics['correct'],
                        'Accuracy': sub_metrics['accuracy']
                    })
            domain_df = pd.DataFrame(domain_rows)
            domain_df.to_csv(domain_output, index=False, encoding='utf-8-sig')
            print(f"Domain-level results saved to: {domain_output}")
    except FileNotFoundError:
        print(f"\nNote: Test file not found ({test_path}). Skipping domain-level evaluation.")
    except Exception as e:
        print(f"\nWarning: Could not perform domain-level evaluation: {e}")

    # Save detailed results
    if results['detailed_results']:
        results_df = pd.DataFrame(results['detailed_results'])
        results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
