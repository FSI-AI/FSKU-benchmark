import re
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='FSKU Benchmark Baseline Inference')
    parser.add_argument(
        '--lang',
        type=str,
        default='kor',
        choices=['kor', 'eng'],
        help='Language of test data (default: kor)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='beomi/gemma-ko-7b',
        help='Model name or path (default: beomi/gemma-ko-7b)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./baseline_submission.csv',
        help='Output file path (default: ./baseline_submission.csv)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=128,
        help='Maximum new tokens to generate (default: 128)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Sampling temperature (default: 0.2)'
    )
    return parser.parse_args()


def extract_question_and_choices(full_text):
    """
    Separates question body and choice list from the full question string.

    Args:
        full_text: Complete question text including choices

    Returns:
        tuple: (question string, list of option strings)
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        # Check if line starts with a number (1-99) indicating a choice
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())

    question = " ".join(q_lines)
    return question, options


def make_prompt(text, lang='kor'):
    """
    Creates a prompt for multiple choice question answering.

    Args:
        text: Full question text with choices
        lang: Language code ('kor' or 'eng')

    Returns:
        str: Formatted prompt string for the model
    """
    question, options = extract_question_and_choices(text)

    if lang == 'kor':
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 문제에 대해 **정답 번호만** 출력하세요.\n\n"
            f"문제: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "정답:"
        )
    else:
        prompt = (
            "You are a financial security expert.\n"
            "For the question below, output **only the correct answer choice number**.\n\n"
            f"Question: {question}\n"
            "Choices:\n"
            f"{chr(10).join(options)}\n\n"
            "Answer:"
        )
    return prompt


def extract_answer_only(generated_text: str, lang: str = 'kor') -> str:
    """
    Post-processes model output to extract only the answer number.

    - Extracts text after "Answer:" or "정답:" delimiter
    - Extracts only the numeric answer from the response
    - Returns "0" if extraction fails or response is empty

    Args:
        generated_text: Raw model output string
        lang: Language code ('kor' or 'eng')

    Returns:
        str: Extracted answer number or "0" if extraction fails
    """
    # Split text based on delimiter
    delimiter = "정답:" if lang == 'kor' else "Answer:"
    if delimiter in generated_text:
        text = generated_text.split(delimiter)[-1].strip()
    else:
        text = generated_text.strip()

    # Return default value for empty responses
    if not text:
        return "0"

    # Extract only the number from the response
    match = re.match(r"\D*([1-9][0-9]?)", text)
    if match:
        return match.group(1)
    else:
        # Return "0" if number extraction fails
        return "0"


def extract_answer_number(answer_text: str) -> str:
    """
    Extracts the answer number from the answer text.

    The answer format in answer.csv is a numeric string (e.g., "2").
    This function handles both numeric-only and legacy "number + text" formats.

    Args:
        answer_text: Answer text from answer.csv

    Returns:
        str: Extracted answer number or "0" if extraction fails
    """
    if pd.isna(answer_text):
        return "0"

    answer_str = str(answer_text).strip()
    match = re.match(r"^\s*([1-9][0-9]?)", answer_str)
    if match:
        return match.group(1)
    return "0"


def evaluate_predictions(predictions: list, ground_truth: pd.Series) -> dict:
    """
    Evaluates predictions against ground truth answers.

    Args:
        predictions: List of predicted answer numbers
        ground_truth: Series of ground truth answers from answer.csv

    Returns:
        dict: Evaluation results containing accuracy and detailed statistics
    """
    # Extract answer numbers from ground truth
    gt_numbers = [extract_answer_number(ans) for ans in ground_truth]

    # Calculate statistics
    total = len(predictions)
    correct = 0
    results = []

    for i, (pred, gt) in enumerate(zip(predictions, gt_numbers)):
        is_correct = (str(pred) == str(gt))
        if is_correct:
            correct += 1
        results.append({
            'question_idx': i,
            'predicted': pred,
            'ground_truth': gt,
            'correct': is_correct
        })

    accuracy = correct / total if total > 0 else 0

    return {
        'total_questions': total,
        'correct_answers': correct,
        'wrong_answers': total - correct,
        'accuracy': accuracy,
        'accuracy_percent': f"{accuracy * 100:.2f}%",
        'detailed_results': results
    }


def print_evaluation_report(eval_results: dict):
    """
    Prints a formatted evaluation report.

    Args:
        eval_results: Dictionary containing evaluation results
    """
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Questions: {eval_results['total_questions']}")
    print(f"Correct Answers: {eval_results['correct_answers']}")
    print(f"Wrong Answers:   {eval_results['wrong_answers']}")
    print(f"Accuracy:        {eval_results['accuracy_percent']}")
    print("=" * 50)

    # Print first 10 wrong answers for debugging
    wrong_answers = [r for r in eval_results['detailed_results'] if not r['correct']]
    if wrong_answers:
        print(f"\nFirst 10 Wrong Answers (out of {len(wrong_answers)}):")
        print("-" * 50)
        for r in wrong_answers[:10]:
            print(f"Q{r['question_idx'] + 1}: Predicted={r['predicted']}, Ground Truth={r['ground_truth']}")


def main():
    # Parse command line arguments
    args = parse_args()

    # Set file paths based on language
    test_path = f'./test({args.lang}).csv'
    answer_path = './answer.csv'

    print(f"=" * 50)
    print(f"FSKU Benchmark Baseline Inference")
    print(f"=" * 50)
    print(f"Language: {args.lang}")
    print(f"Model: {args.model}")
    print(f"Test file: {test_path}")
    print(f"Output file: {args.output}")
    print(f"=" * 50)

    # Load test data and answer data
    test = pd.read_csv(test_path)
    answer = pd.read_csv(answer_path)

    print(f"Loaded {len(test)} questions")

    # Load tokenizer and model (4-bit quantization)
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )

    # Create inference pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # Run inference on all questions
    preds = []

    for q in tqdm(test['Question'], desc="Inference"):
        prompt = make_prompt(q, lang=args.lang)
        output = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9
        )
        pred_answer = extract_answer_only(output[0]["generated_text"], lang=args.lang)
        preds.append(pred_answer)

    # Save results to submission file
    submission = pd.DataFrame({'Answer': preds})
    submission.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nPredictions saved to: {args.output}")

    # Evaluate predictions against ground truth
    eval_results = evaluate_predictions(preds, answer['Answer'])

    # Print evaluation report
    print_evaluation_report(eval_results)

    # Save detailed results to CSV for further analysis
    results_path = args.output.replace('.csv', '_details.csv')
    results_df = pd.DataFrame(eval_results['detailed_results'])
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
