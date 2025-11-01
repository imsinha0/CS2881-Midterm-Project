from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score

class Evaluator:
    def __init__(self, predictions, references):
        self.predictions = predictions
        self.references = references

    def evaluate(self):
        # ROUGE-L
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = rouge.score(self.references, self.predictions)
        rouge_l = rouge_scores["rougeL"].fmeasure

        # Tokenize
        reference_tokens = word_tokenize(self.references)
        prediction_tokens = word_tokenize(self.predictions)

        # BLEU
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)

        # Token-level F1 (unordered overlap)
        ref_set, pred_set = set(reference_tokens), set(prediction_tokens)
        tp = len(ref_set & pred_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        token_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # BERTScore
        _, _, F1 = bert_score([self.predictions], [self.references], lang="en", verbose=False)
        bert_f1 = F1[0].item()

        return {
            "rougeL": rouge_l,
            "bleu": bleu,
            "token_f1": token_f1,
            "bertscore_f1": bert_f1
        }
