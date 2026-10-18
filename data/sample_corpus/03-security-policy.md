# Lanternfish Analytics — Information Security Policy

*Fictional company. All names, numbers and policies are invented for the RAG demo.*

## 1. Accounts and passwords

All company accounts use single sign-on through our identity provider. Passwords must be at least **14 characters** long and are stored in the company password manager. **Multi-factor authentication is mandatory** on every account; hardware security keys are required for anyone with production access.

Shared accounts are not allowed. Access is reviewed every **quarter** by each team lead, and access for leavers is removed on their last working day.

## 2. Devices

Company laptops must have full-disk encryption turned on, automatic screen lock after **5 minutes** of inactivity, and the endpoint protection agent installed. Personal devices may only be used for email and chat through the managed mobile app; they must never store customer data.

## 3. Data classification

| Class | Examples | Rules |
|---|---|---|
| Public | Website, published docs | No restrictions |
| Internal | Roadmaps, handbook | Share inside the company only |
| Confidential | Contracts, salaries | Need-to-know, encrypted at rest |
| Restricted | Customer datasets, credentials | Production access only, never copied to laptops |

Customer data is **Restricted** by default.

## 4. Reporting a security incident

Report suspected incidents — phishing that you clicked, a lost laptop, data sent to the wrong person — in the **#security-incidents** channel or to security@lanternfish.example **within 1 hour** of noticing. Do not try to investigate on your own. The on-call security engineer will acknowledge within **15 minutes** during business hours and within 1 hour at other times.

Lost or stolen devices are wiped remotely once reported.

## 5. New software and vendors

Any new SaaS tool that will hold Internal data or above needs a vendor security review before purchase. The review normally takes **5 working days**. Browser extensions must be on the approved list.
