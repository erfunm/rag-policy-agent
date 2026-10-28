---
policy_id: POL-SAN-004
title: Sanctions Screening Policy
version: "1.3"
effective_date: 2023-09-01
owner: Head of Financial Crime Compliance
acl_roles: [support, operations, compliance]
summary: Customer and payment screening against sanctions lists, match handling and freezing.
---
# Sanctions Screening Policy (v1.3)

## 1. What is screened
Customers are screened against consolidated sanctions lists at onboarding and again every night. Every outgoing and incoming payment is screened in real time before it is released.

## 2. Potential matches
The screening engine flags a potential match when the name similarity score is 85% or higher. A payment with a potential match is held automatically. The Sanctions team must review a held payment within 4 business hours.

## 3. Confirmed matches
If a match is confirmed, the funds are frozen and must not be returned to the sender. The Sanctions team reports the frozen funds to the national sanctions authority within 1 business day. Customer-facing staff must tell the customer only that the payment is under review.

## 4. False positives
A false positive can be discounted only by a Sanctions analyst, who records the reason. Discounted names are added to the good-guy list, which is reviewed every 6 months.
