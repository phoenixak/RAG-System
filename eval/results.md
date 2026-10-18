# Evaluation results

Corpus: 8 documents, 50 chunks. Golden set: 42 answerable + 4 unanswerable questions.

## Retrieval (does the right document come back?)

| Retriever | Hit@1 | Recall@3 | MRR | p50 latency (ms) |
|---|---:|---:|---:|---:|
| Vector only (bge-small) | 88% | 100% | 0.929 | 11 |
| BM25 only | 93% | 100% | 0.960 | 0 |
| Hybrid (RRF) | 90% | 100% | 0.952 | 11 |
| Hybrid + cross-encoder rerank | 98% | 100% | 0.988 | 123 |

## Answers (llama.cpp · Qwen3-1.7B-Q4_K_M.gguf, CPU)

| Metric | Value |
|---|---:|
| Answer contains the expected fact | 95% |
| Every citation points to the right document | 100% |
| Refused when the answer is not in the docs | 100% |
| False refusals (answer was in the docs) | 0% |
| Median time to first token (s) | 1.4 |
| Median full answer time (s) | 2.2 |

<details><summary>Every answer</summary>

| Question | Answer | OK |
|---|---|---|
| What are the core collaboration hours? | The core collaboration hours are 10:00 to 15:00 in your local time zone, Monday to Thursday. This is supported by source [1]. | ✓ |
| Can I book an internal meeting on a Friday? | If you book an internal meeting on a Friday, it may be allowed depending on the severity of the issue, but the documents do not explicitly state whether internal meetings are permitted on Fridays. [1] | ✗ |
| How is overtime compensated? | Overtime must be agreed with your manager in advance and is compensated as time off in lieu at a rate of 1.5 hours per extra hour worked [1]. | ✓ |
| How long can I work from another country without approval? | Working from another country for more than 30 consecutive days requires written approval [1]. | ✓ |
| How much is the home-office allowance? | The home-office allowance is USD 800 in the first month, plus USD 40 per month towards internet costs. [1] | ✓ |
| How many days of annual leave do employees get? | Annual leave is 25 working days per calendar year, with up to 5 unused days carried over to the next year and must be used by 31 March [1]. | ✓ |
| How many unused vacation days can be carried over? | The number of unused vacation days that can be carried over is up to 5 days [1]. | ✓ |
| When do I need a doctor's note for sick leave? | A doctor's note is only required for absences longer than 3 consecutive days [1]. | ✓ |
| How long is parental leave for the primary caregiver? | Parental leave for the primary caregiver is 20 weeks fully paid [1]. | ✓ |
| What is the yearly learning budget? | The yearly learning budget is USD 1,500 for each employee. [1] | ✓ |
| How often are laptops refreshed? | Laptops are refreshed every three years [1]. | ✓ |
| How long do I have to submit an expense claim? | Expenses must be submitted in the finance app within 60 days of the purchase date [1]. | ✓ |
| Who has to approve an expense claim of 3,000 dollars? | The expense claim of 3,000 dollars requires approval from both the line manager and the Finance Director [1]. | ✓ |
| What is the limit per person for client entertainment? | The limit per person for client entertainment is up to USD 90, and the names of the attendees must be listed on the claim [1]. | ✓ |
| Can I expense a parking fine? | Parking fines are not reimbursable [1]. | ✓ |
| How quickly must corporate card statements be reconciled? | Corporate card statements must be reconciled within 15 days of the statement date [1]. | ✓ |
| What is the minimum password length? | The minimum password length is 14 characters [1]. | ✓ |
| Who needs a hardware security key? | Hardware security keys are required for anyone with production access [1]. | ✓ |
| How soon must a security incident be reported? | The security incident must be reported within 1 hour of noticing [1]. | ✓ |
| After how many minutes must a laptop lock automatically? | The laptop must lock automatically after 5 minutes of inactivity [1]. | ✓ |
| How is customer data classified? | Customer data is classified as **Restricted** by default [1]. | ✓ |
| How long does a vendor security review take? | The vendor security review takes 5 working days [1]. | ✓ |
| What is the refund window for annual plans? | Annual plans can be refunded in full within 30 days of the start date [1]. | ✓ |
| Do you refund a partial month on a monthly plan? | No, you do not refund a partial month on a monthly plan. This is supported by source [1]. | ✓ |
| What fee applies when cancelling an annual plan after 30 days? | The fee applied when cancelling an annual plan after 30 days is 10% of the remaining amount [1]. | ✓ |
| How long does it take to receive a refund? | Refunds are paid to the original payment method within 14 days [2]. | ✓ |
| Are enterprise contracts covered by the refund policy? | Enterprise contracts (more than 250 seats or with a custom DPA) are not covered by the refund policy [1]. | ✗ |
| What is the uptime commitment for Beacon? | The uptime commitment for Beacon is **99.9%** of the time in every calendar month [1]. | ✓ |
| What service credit do I get if uptime falls below 95 percent? | The service credit for uptime below 95 percent is 50% of the month's fee [1]. | ✓ |
| What is the first response time for a Sev 1 issue on the Enterprise plan? | The first response time for a Sev 1 issue on the Enterprise plan is 15 minutes, 24/7 [2]. | ✓ |
| When are maintenance windows? | Maintenance windows are on Sundays between 02:00 and 05:00 UTC [1]. | ✓ |
| When can I fly business class? | Business class can only be booked on flights longer than 10 hours with the Finance Director's approval [1]. | ✓ |
| What is the hotel cap in Lisbon? | The hotel cap in Lisbon is USD 180 [1]. | ✓ |
| How much is the daily per diem in London? | The daily per diem in London is USD 95 [1]. | ✓ |
| What is the mileage rate for using my own car? | The mileage rate for using your own car is reimbursed at USD 0.55 per kilometre [1]. | ✓ |
| When does security training have to be completed? | Security training must be completed within the first 5 working days [1]. | ✓ |
| How long is the probation period? | The probation period is 3 months [1]. | ✓ |
| How long is raw event data kept on the Business plan? | Raw event data is kept for 13 months on the Business plan [1]. | ✓ |
| Which plan includes SAML single sign-on? | SAML single sign-on is available on the Enterprise plan [1]. | ✓ |
| What is the query API rate limit? | The query API rate limit is 60 requests per minute per API key [1]. | ✓ |
| Can I change my workspace data region later? | Data never leaves the chosen region, and the region cannot be changed after the workspace is created [1]. | ✓ |
| What discount do I get for annual billing? | The 15% discount applies to Starter and Business plans for annual billing [1]. | ✓ |
| What is the company's stock ticker symbol? | I couldn't find that in the documents. | ✓ |
| How many employees does Lanternfish have in Tokyo? | I couldn't find that in the documents. | ✓ |
| Does the gym membership get reimbursed? | I couldn't find that in the documents. | ✓ |
| What is the dress code for client meetings? | I couldn't find that in the documents. | ✓ |

</details>
