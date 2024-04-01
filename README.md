# Wikipedia-Category-Classification
## Demo link
[Wiki-Category-Classification Demo](https://huggingface.co/spaces/khoavpt/demo-app?fbclid=IwAR1xXSg5BjWQUSrl8TP951pwHg01tmtFU3w9M7moGRnSvkPMMu1LU91ofng_aem_AUog7kSyu_A7-BJZHqig88VCbn1cb-KcpQUz61RPI_y4QjMdwfipjollGj89b9W0dRG44T6AUMUYNHifFEvsGHwP)

## Experiment results
### Table 1: Performance of Different Topic Classification Models

| Category | Model | Accuracy | Macro F1 |
|---|---|---|---|
| Machine Learning | SVM | 86.00% | 0.8489 |
| | Random Forest | 79.00% | 0.7681 |
| | Gradient Boosting | 78.00% | 0.7556 |
| | KNN | 85.50% | 0.8435 |
| RNN based | DeepRNN | 61.50% | 0.5309 |
| | BiGRU | 87.25% | 0.8635 |
| | BiLSTM | 85.75% | 0.8476 |
| | BiLSTM+Attention | 85.25% | 0.8412 |
| CNN based | TextCNN | 85.00% | 0.8364 |
| | CharCNN | 66.25% | 0.5713 |
| Transformer | Pho-BERT | 86.00% | 0.8505 |
| | XLM-RoBERTa-VN | 87.00% | 0.8624 |

### Table 2: Accuracy of Different Topic Classification Models on All Four Labels

| Model | Natural Sciences | Social Sciences | Engineering | Culture |
|---|---|---|---|---|
| SVM | 90.77% | 85.59% | 86.67% | 77.92% |
| Random Forest | 81.12% | 87.50% | 77.97% | 62.79% |
| Gradient Boosting | 83.33% | 81.82% | 75.00% | 70.13% |
| KNN | 87.97% | 85.75% | 85.71% | 80.28% |
|---|---|---|---|---|
| DeepRNN | 93.33% | 60.83% | 7.50% | 68.75% |
| BiGRU | 95.83% | 85.00% | 85.00% | 80.00% |
| BiLSTM | 97.50% | 83.33% | 77.50% | 80.00% |
| BiLSTM+Attention | 95.00% | 84.17% | 87.50% | 70.00% |
| TextCNN | 97.50% | 85.00% | 76.25% | 75.00% |
| CharCNN | 93.33% | 85.00% | 8.75% | 55.00% |
|---|---|---|---|---|
| Pho-BERT | 95.80% | 91.43% | 79.31% | 74.16% |
| XLM-RoBERTa-VN | 98.26% | 93.14% | 79.55% | 76.14% |

## Demo screenshots
### Text input
![](img/demo_screenshot2.png)

### Link input
![](img/demo_screenshot1.png)
