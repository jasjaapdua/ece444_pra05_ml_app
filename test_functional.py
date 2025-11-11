"""
ECE444 PRA05 - Functional Tests for Fake News Detector
Author: Jasjaap Dua

This test suite checks that the deployed Flask REST API
correctly classifies known examples as FAKE or REAL.
"""

import pytest
import requests

BASE_URL = "http://ece444-pra05-env.eba-vraaipee.us-east-2.elasticbeanstalk.com/predict"

TEST_CASES = [
    ("Eiffel Tower washes up on Delaware Beach.", "FAKE"),
    ("University of Toronto moved to Montreal", "FAKE"),
    ("The Prime Minister announced new economic policies", "REAL"),
    ("Syria's President Meets Trump at White House for First Time", "REAL"),
]


@pytest.mark.parametrize("text,expected", TEST_CASES)
def test_prediction_correctness(text, expected):
    """
    Each test case sends one message to the /predict API
    and asserts that the returned label matches the expected one.
    """
    resp = requests.post(BASE_URL, json={"message": text})
    assert resp.status_code == 200, f"Bad status code {resp.status_code} for: {text}"

    data = resp.json()
    actual = str(data.get("label", "")).strip().upper()

    print(f"\nInput: {text}\nExpected: {expected}, Got: {actual}")
    assert actual == expected, f"Expected {expected} but got {actual} for input: {text}"


def test_invalid_input_handling():
    """
    Ensure the API properly rejects missing input payloads.
    """
    resp = requests.post(BASE_URL, json={})
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    data = resp.json()
    assert "error" in data, "Missing error message for invalid input"
