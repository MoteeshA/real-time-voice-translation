# English accent control

XTTS conditions speaker identity using the voice reference and conditions the output language using a language token. It does not expose a dedicated Indian-English accent control.

Li et al. evaluated an experimental approximation: keep the text in English and use XTTS's Hindi (`hi`) language token. Their XTTS baseline showed higher Indian-accent similarity for this setting (0.459 vs. 0.270) but substantially worse word error rate (23.2% vs. 8.6%). In this app, a user trial sounded worse, so the approximation has been removed from the interface.

The app uses the standard English (`en`) token. For Indian-accented English, use a reference recorded by the same speaker speaking English in that accent. Hindi-only reference speech can condition speaker timbre but does not reliably specify English pronunciation or prosody. Fine-tuning with licensed Indian-English data is the research direction for reliable accent control.

Citation: [Li et al. (2025), arXiv:2508.07426](https://arxiv.org/abs/2508.07426).
