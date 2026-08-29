# Security

## The default assumes localhost

No authentication, no rate limit unless configured, and the API can delete models from disk.
That is deliberate for one person on one machine. Turn both on in `config.toml` before the
daemon listens on anything else - the listening address is the security boundary, because inside
it there is none.

Prompts never leave the machine. What reaches the network: fetching a model you asked for, and
cluster nodes announcing themselves if configured.

## Reporting

Open a private security advisory on the repository. If that is unavailable, open an issue saying
only that you found something and asking where to send it - no working exploit in public.

Include what you did, what happened, what you expected. A proof of concept beats a description;
a commit hash beats "latest".

No bounty, no response-time commitment. Reports are read and answered when seen.

## Scope

**In:** anything reaching past what the API should expose - reading or writing outside the
configured model and cache directories, executing code from a model file or a prompt, or making
network requests it was not asked to make.

**Out:** the unauthenticated default above, resource exhaustion with no rate limit configured,
and model output. What a model generates is the model's; no filtering is claimed here.
