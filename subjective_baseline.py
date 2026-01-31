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
    parser = argparse.ArgumentParser(description='FSKU Subjective Benchmark Baseline Inference')
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
        default='./subjective_submission.csv',
        help='Output file path (default: ./subjective_submission.csv)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum new tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (default: 0.3)'
    )
    return parser.parse_args()


def make_prompt(question: str, lang: str = 'kor') -> str:
    """
    Creates a prompt for subjective question answering.

    Args:
        question: The question text
        lang: Language code ('kor' or 'eng')

    Returns:
        str: Formatted prompt string for the model
    """
    if lang == 'kor':
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 핵심 내용을 포함하여 간결하고 정확하게 답변하세요.\n\n"
            f"질문: {question}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "You are a financial security expert.\n"
            "Answer the following question concisely and accurately, including key points.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    return prompt


def extract_answer(generated_text: str, lang: str = 'kor') -> str:
    """
    Post-processes model output to extract only the answer.

    Args:
        generated_text: Raw model output string
        lang: Language code ('kor' or 'eng')

    Returns:
        str: Extracted answer text
    """
    # Split text based on delimiter
    delimiter = "답변:" if lang == 'kor' else "Answer:"
    if delimiter in generated_text:
        text = generated_text.split(delimiter)[-1].strip()
    else:
        text = generated_text.strip()

    # Remove any trailing special tokens or artifacts
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML-like tags
    text = text.strip()

    return text


def main():
    # Parse command line arguments
    args = parse_args()

    # Set file paths based on language
    test_path = f'./subjective_test({args.lang}).csv'

    print("=" * 60)
    print("FSKU Subjective Benchmark Baseline Inference")
    print("=" * 60)
    print(f"Language: {args.lang}")
    print(f"Model: {args.model}")
    print(f"Test file: {test_path}")
    print(f"Output file: {args.output}")
    print("=" * 60)

    # Load test data
    test = pd.read_csv(test_path)
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
    predictions = []

    for question in tqdm(test['Question'], desc="Inference"):
        prompt = make_prompt(question, lang=args.lang)
        output = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True
        )
        answer = extract_answer(output[0]["generated_text"], lang=args.lang)
        predictions.append(answer)

    # Save results to submission file
    submission = pd.DataFrame({
        'Question': test['Question'],
        'Predicted_Answer': predictions
    })
    submission.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\nPredictions saved to: {args.output}")

    # Print sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions (First 3)")
    print("=" * 60)
    for i in range(min(3, len(predictions))):
        print(f"\nQ{i+1}: {test['Question'].iloc[i][:50]}...")
        print(f"A{i+1}: {predictions[i][:100]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()
