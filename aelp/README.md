# AELP is implemented as APEX

The survey's AELP refers to Hsieh et al., [Automatic Engineering of Long
Prompts](https://aclanthology.org/2024.findings-acl.634/), arXiv:2311.10117.
The final paper names its algorithm APEX (Automated Prompt Engineering Xpert).
Use the existing [`apex` implementation](../apex/README.md): it implements
sentence-level beam search, LinUCB sentence selection, and history-guided
rewriting. A second optimizer under this directory would duplicate that method.
This identification was checked against the final ACL paper, not inferred from
the acronym. No separate official AELP repository was found in the paper or
the survey's linked bibliography.
