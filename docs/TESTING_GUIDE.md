# 🧪 Testing & Debugging Guide — novice → mastery

This starter is built for learning, and testing is where a lot of that learning
happens. This guide takes you from *"what is a test?"* to writing professional
regression tests — using [`tests/test_stubs.py`](../tests/test_stubs.py) as your
graded practice ladder. (New to the project? Read the
[Beginner's Guide](BEGINNERS_GUIDE.md) first.)

---

## 1. What is a test?

A tiny piece of code that automatically checks *"does this behave the way I expect?"*.

```python
def test_addition():
    assert 2 + 2 == 4     # if false, the test FAILS and points at this line
```

`assert X` = "I claim X is true." Tests let you change code *without fear*: run them
and know in seconds whether you broke anything.

## 2. Running the tests

```bash
pip install -r requirements.txt && pip install -e .
pytest -v              # verbose: one line per test
pytest -k drift        # only tests whose name contains "drift"
pytest -x              # stop at the first failure
```

Dots (`.`) pass, `s` = skipped (the stubs, waiting for you), `F` = fail (with a
traceback pointing at the problem).

## 3. Arrange, Act, Assert — the shape of every test

```python
def test_clean_uppercases():
    df = pd.DataFrame({"make": [" maruti "], "model": ["swift"], "selling_price": [5.0]})  # Arrange
    out = data.clean(df)                                                                    # Act
    assert out["make"].iloc[0] == "MARUTI"                                                  # Assert
```

Memorise **A-A-A** and you can write almost any test.

## 4. The difficulty ladder (in `tests/test_stubs.py`)

The stubs are grouped from easy to advanced:

| Group | Focus | Skill it builds |
| :---- | :---- | :-------------- |
| 🟢 **Data** | cleaning, pandas basics | how to run pytest; Arrange-Act-Assert |
| 🟡 **Features** | target encoding, band edges | testing pure functions |
| 🟠 **Serving** | reject an unknown make/model | edge cases, `pytest.raises` |
| 🔴 **Modelling** | a tuned model beats the baseline | integration + hyper-parameter search |
| 🟣 **Mastery** | `@parametrize`, domain *properties* | professional testing |
| 🐞 **Debugging drills** | turn a real past bug into a regression test | the core debugging loop |

**To complete a stub:** delete its `@pytest.mark.skip(...)`, replace `pytest.fail("TODO…")`
with real `assert`s, and run `pytest`.

## 5. Debugging when a test goes red

1. **Read the traceback bottom-up** — the last lines say *what* failed and *where*.
2. **Reproduce small:** `pytest tests/test_stubs.py::test_name -x`.
3. **Print or breakpoint:** add `print(value)` (run with `pytest -s`) or `breakpoint()`.
4. **Question the test, not just the code** — often the *expectation* is wrong.
5. **Fix, re-run, keep the test.** It now guards that behaviour forever — a *regression test*.

## 6. Troubleshooting cheat-sheet

| Symptom | Likely cause & fix |
| :------ | :----------------- |
| `ModuleNotFoundError: car_pricing` | Run `pip install -e .`, or run pytest from the repo root. |
| `FileNotFoundError: price_pipeline.pkl` | Run `python -m car_pricing.train` once. |
| A stub "passes" doing nothing | You deleted the skip but left `pytest.fail` / no asserts. |
| Test writes files into the repo | Use `tmp_path` + `monkeypatch` to redirect paths. |
| Flaky test (passes sometimes) | You depend on randomness/order — fix a seed or isolate state. |

## 7. The mastery mindset

Two habits separate a novice from a pro:

1. **Make code testable.** Keep logic in small pure functions (this package already does),
   so it can be checked in isolation.
2. **Every bug you fix leaves a test behind.** The 🐞 debugging drills practise exactly this —
   e.g. the *unknown-make silent fallback* (make the API reject it, then test that it does)
   and the *servable-model* regression. Do this and your code only ever gets more reliable.

> Looking for the *what*-to-build next? Each stub maps to a real MLOps extension — the
> fully-worked versions live in the sibling
> **[app_car_prices_mlops](https://github.com/kalyancn4u/app_car_prices_mlops)** repo, so
> you can attempt a stub here, then compare with a reference implementation there.
