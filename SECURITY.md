# Security

## Default

The default configuration assumes localhost.  
It has no authentication and no rate limit, and the API can delete models from disk.
That suits 1 person on 1 machine.

Before the daemon listens on any other address, set `require_auth`, `api_keys` and `rate_limit_per_minute` under `[server]` in `config.toml`.  
The listening address is the security boundary.
Inside it there is none.

Prompts never leave the machine.  
The network is reached to fetch a model you asked for, and by cluster nodes announcing themselves when configured.

## Reporting

Open a private security advisory on the repository.
If that is unavailable, open an issue that says only that you found something and asks where to send it.  
Publish no working exploit.

Include:

- What you did, what happened, what you expected.
- A proof of concept, which beats a description.
- A commit hash, which beats "latest".

There is no bounty and no response-time commitment.
Reports are read and answered when seen.

## Scope

| In scope | Out of scope |
|---|---|
| Anything that reaches past what the API should expose | The unauthenticated default |
| Reading or writing outside the configured model and cache directories | Resource exhaustion with no rate limit configured |
| Executing code from a model file or a prompt | Model output: what a model generates is the model's, and no filtering is claimed |
| Network requests the server was not asked to make | |
