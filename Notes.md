# [AI] Notes

## How to Train a Large Language Model
1. **Data Collection**: Collect a large dataset of text.
2. **Preprocessing**: Tokenize the text into subwords.
3. **Model Architecture**: Choose a model architecture. [BPE](Glossory.md#bpe-byte-pair-encoding) algorithm for tokenization.
4. **Training**: Train the model on the dataset. Data splitting and training strategies.
5. **Fine-Tuning**: Fine-tune the model on a smaller dataset.
6. **Inference**: Use the model to generate text.



## Training Process (example with Deepseek)
1. Train V3-Base with RL for reasoning -> R1-Zero
2. Create SFT data from R1-Zero using rejection sampling + synthetic data from V3.
3. Re-train V3-Base from scratch on SFT data., followed by RL (reasoning + human prefs.)

### GRPO: Group Relative Policy Optimization
- For each input question, sample mulitple responses
- Compute reward for each, and calculate its **group-normailzed** advantage
- No need for critic model (answers are rewarded compared to the group)


## Deepseek Evolution
Foundational Models
| Model   | Parameters | Methods                                                                                                                                        | Notes                              |
|---------|------------|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| V1      | 67B        | Traditional Transformer                                                                                                                        |                                    |
| V2      | 236B       | (MHLA) Multi-head latent Attention; (MoE) Mixture of Experts                                                                                   | made the model fast                |
| V3      | 671B       | (RL) Reinforcement Learning; (MHLA) Multi-head latent Attention; (MoE) Mixture of Experts                                                      | Balance load amongst many GPU's    |
| R1-Zero | 671B       | (RL) Reinforcement Learning; (MHLA) Multi-head latent Attention; (MoE) Mixture of Experts; (CoT) Chain of Thougt                               | Reasoning model                    |
| R1      | 671B       | (RL) Reinforcement Learning; (SPT) Supervised Fine-Tuning; (MHLA) Multi-head latent Attention; (MoE) Mixture of Experts; (CoT) Chain of Thougt | Reasoning model; Human preferences |


## Transformer-Based Language Models
| Model      | Full Name                                                                             | Organization |
|------------|---------------------------------------------------------------------------------------|--------------|
| BERT       | Bidirectional Encoder Representations from Transformers                               | Google       |
| XLNet      | Generalized Autoregressive Pretraining for Language Understanding                     | Google/CMU   |
| RoBERTa    | A Robustly Optimized BERT Approach                                                    | Facebook     |
| DistilBERT | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter         | Hugging Face |
| CTRL       | Conditional Transformer Language Model for Controllable Generation                    | Salesforce   |
| GPT-2      | Generative Pre-trained Transformer                                                    | OpenAI       |
| ALBERT     | A Lite BERT for Self-supervised Learning of Language Representations                  | Google       |
| Megatron   | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | NVIDIA       |