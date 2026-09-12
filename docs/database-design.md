# Lifelink-AI Database Design

## Users

- user_id
- name
- email
- password_hash
- role
- hospital_id
- created_at

## Hospitals

- hospital_id
- hospital_name
- address
- latitude
- longitude
- contact_number

## Donors

- donor_id
- hospital_id
- age
- weight
- blood_type
- medical_approval
- status
- created_at

## Recipients

- recipient_id
- hospital_id
- age
- weight
- bmi
- blood_type
- diagnosis
- biological_markers
- waiting_time
- status
- created_at

## Organs

- organ_id
- donor_id
- organ_type
- health_score
- condition
- availability_status
- tracking_id
- scanned_at

## Match Results

- match_id
- donor_id
- recipient_id
- compatibility_score
- model_name
- model_version
- rank
- created_at

## Doctor Decisions

- decision_id
- match_id
- doctor_id
- decision
- reason
- decided_at

## Allocations

- allocation_id
- match_id
- hospital_id
- allocation_status
- allocated_at

## Transport Routes

- route_id
- allocation_id
- source_hospital
- destination_hospital
- distance
- estimated_time
- route_algorithm
- route_status

## Audit Logs

- log_id
- user_id
- action
- entity
- entity_id
- timestamp