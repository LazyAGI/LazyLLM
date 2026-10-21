import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import lazyllm
from lazyllm.tools import data  # noqa: F401


@pytest.fixture
def scorer():
    torch.manual_seed(21)
    tokens = ['[UNK]', 'The', 'capital', 'of', 'France', 'is', 'Paris', '.', 'Answer', ':']
    tokenizer = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    model_config = GPT2Config(
        vocab_size=len(tokens), n_positions=64, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )
    model = GPT2LMHeadModel(model_config)
    model.eval()
    return lazyllm.data.Text2qa.IFDScorer(model, tokenizer, _save_data=False, _max_workers=1)


def causal_answer_loss(scorer, context, answer):
    ids = scorer.tokenizer.encode(context + answer, return_tensors='pt')
    start = max(len(scorer.tokenizer.encode(context)), 1)
    labels = ids.clone()
    labels[:, :start] = -100
    with torch.no_grad():
        return float(scorer.model(ids, labels=labels).loss)


def test_registered_operator_matches_native_causal_model_loss(scorer):
    query, answer = 'The capital of France is ', 'Paris .'
    result = scorer([{'query': query, 'answer': answer}])[0]
    expected_cas = causal_answer_loss(scorer, query, answer)
    expected_das = causal_answer_loss(scorer, '', answer)
    assert result['CAS'] == pytest.approx(expected_cas)
    assert result['DAS'] == pytest.approx(expected_das)
    assert result['IFD_score'] == pytest.approx(expected_cas / expected_das)


def test_full_token_loss_matches_model_native_shifted_loss(scorer):
    ids = scorer.tokenizer.encode('The capital of France is Paris .', return_tensors='pt')
    with torch.no_grad():
        expected = float(scorer.model(ids, labels=ids).loss)
    observed = scorer._get_token_loss('', 'The capital of France is Paris .')
    assert np.mean(observed) == pytest.approx(expected)


def test_one_unconditional_token_has_no_next_token_target(scorer):
    assert scorer._calc_loss_part('', 'Paris') is None
