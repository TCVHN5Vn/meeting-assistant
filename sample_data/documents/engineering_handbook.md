# Northwind Engineering Handbook

## Deployment Process

All production deployments go through the release pipeline. There are no
manual deploys to production under any circumstances, including hotfixes.

1. Open a pull request against `main`. CI runs unit tests, integration
   tests, and the security scan automatically.
2. Obtain at least one approving review. Changes touching the payments
   service or the authentication service require two approvals, one of
   which must come from the owning team.
3. Merge to `main`. This triggers an automatic deploy to the staging
   environment within roughly four minutes.
4. Verify the change on staging. The smoke test suite runs automatically,
   but the author is expected to manually confirm the specific behaviour
   they changed.
5. Promote to production using the `deploy promote` command. Promotion is
   gated on a green staging run no older than two hours.

Production deploys are frozen from 16:00 Friday until 09:00 Monday, and for
the full week around any major customer launch. The freeze can only be
lifted by the on-call engineering manager, and only for a Sev-1 incident.

## Rollback

If a deploy causes an incident, roll back first and investigate afterwards.
The `deploy rollback` command reverts to the previous known-good build and
typically completes in under ninety seconds. Do not attempt to fix forward
during an active incident unless the on-call lead explicitly agrees.

## Code Review Standards

Reviews are expected within one working day. A reviewer should be able to
understand the change from the pull request description alone; if they
cannot, that is a defect in the description, not in the reviewer.

Reviewers check for: correctness, test coverage of the new behaviour,
backwards compatibility of any API or database change, and whether the
change is the smallest one that solves the problem. Style is not reviewed
by humans; the formatter is the authority and its output is not up for
discussion.

Pull requests over 400 lines of diff should be split. Large reviews receive
disproportionately shallow attention, which defeats the purpose.

## On-Call

Engineering runs a weekly on-call rotation, handed over at 10:00 on
Wednesdays. The on-call engineer is the first responder for all Sev-1 and
Sev-2 alerts and is expected to acknowledge a page within fifteen minutes
during working hours and within thirty minutes outside them.

On-call engineers are not assigned sprint work. The rotation is compensated
at a flat rate of 400 EUR per week, plus time off in lieu for any incident
handled outside working hours.

Severity definitions:
- Sev-1: customer-facing outage or data loss. Page immediately, all hands.
- Sev-2: significant degradation with a workaround. Page during hours.
- Sev-3: minor issue, no customer impact. File a ticket, no page.

## Testing Requirements

New code requires unit tests. Bug fixes require a regression test that
fails before the fix and passes after it. Integration tests are required
for any change that crosses a service boundary.

The build fails below 70 percent line coverage on changed files. Coverage
of the codebase as a whole is deliberately not a target, because optimising
it produces tests that assert nothing.
