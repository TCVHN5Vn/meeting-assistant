# Northwind Product Roadmap 2026

## Q3 2026 — Reliability

The theme for Q3 is reliability. We ended Q2 at 99.4 percent availability
against a 99.9 percent target, and customer escalations were dominated by
timeouts in the search service rather than by missing features.

Committed work:
- Rewrite the search indexing pipeline to remove the nightly full rebuild.
  The rebuild is the direct cause of the 03:00 latency spike and of three
  of the last five Sev-1 incidents.
- Introduce per-tenant rate limiting on the public API. A single customer's
  batch job is currently able to degrade service for everyone.
- Complete the migration off the legacy job scheduler. This has slipped
  twice and is now blocking the observability work.

Explicitly not in Q3: the reporting redesign, mobile offline mode, and SSO
for the free tier. These were considered and deferred.

## Q4 2026 — Enterprise Readiness

Q4 shifts to what enterprise prospects have blocked on in deals.

- SAML and SCIM support. This is the single most frequently cited blocker
  in lost enterprise deals, appearing in 11 of the 17 losses last quarter.
- Audit log export, retained for seven years, with a documented schema.
- Data residency in the EU, with a customer-selectable region at signup.
- Role-based access control with custom roles, replacing the current fixed
  three-tier model of admin, member, and viewer.

## Success Metrics

- Availability of 99.9 percent or better, measured monthly at the edge.
- p95 search latency under 400 ms, down from the current 1.2 seconds.
- Enterprise deal cycle reduced from 94 days to under 70.
- Support ticket volume per active account reduced by 30 percent.

## Known Risks

The search rewrite depends on the new storage layer, which is owned by a
team of two and is already the critical path for two other projects. If it
slips, Q3 slips with it, and there is no realistic way to parallelise
around it.

Data residency requires legal review in each target jurisdiction, and legal
capacity is constrained until October. Starting that review in Q3 rather
than Q4 is the only mitigation available.
