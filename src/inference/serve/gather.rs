//! Requests that share a key run as one batch behind the group ahead of them.
//!
//! A request whose key has no group running starts at once, as a group of one, and its
//! runner is a task of its own. Requests of that key that arrive while a group runs queue
//! behind it. When the running group ends, its runner takes the queue whole as the next
//! group, up to the cap the first queued request asked for, and the rest wait for the
//! group after. A lone request never waits, and a group never holds more jobs than that
//! cap.
//!
//! Each answer goes back to the request that brought the job, so a request whose client
//! goes away does not take its companions' answers with it.

use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

/// One group's work: the jobs in arrival order to one answer per job, in the same order.
type Runner<J, R> = Box<dyn FnOnce(Vec<J>) -> Pin<Box<dyn Future<Output = Vec<R>> + Send>> + Send>;

/// A request in a queue: its job, where its answer goes, how it runs a group and how
/// many jobs a group it opens may hold.
struct Member<J, R> {
    job: J,
    answer: oneshot::Sender<R>,
    run: Runner<J, R>,
    cap: usize,
}

/// The keys with a group running, each with the requests queued behind it.
type Queues<K, J, R> = Arc<Mutex<HashMap<K, Vec<Member<J, R>>>>>;

/// The running groups, one per key, and their queues.
pub struct Gather<K, J, R> {
    queues: Queues<K, J, R>,
}

impl<K, J, R> Clone for Gather<K, J, R> {
    fn clone(&self) -> Self {
        Self {
            queues: self.queues.clone(),
        }
    }
}

impl<K, J, R> Default for Gather<K, J, R> {
    fn default() -> Self {
        Self {
            queues: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<K, J, R> Gather<K, J, R>
where
    K: Eq + Hash + Clone + Send + 'static,
    J: Send + 'static,
    R: Send + 'static,
{
    /// Run `job` at once when no group of `key` runs, or with the requests of `key` that
    /// queue behind the running group, at most `cap` of them together.
    ///
    /// `run` takes the jobs in arrival order and answers one result per job, in the same
    /// order; the first job of a group runs it for the whole group. `None` when the
    /// runner ended without an answer for this job.
    pub async fn submit<F, Fut>(&self, key: K, job: J, cap: usize, run: F) -> Option<R>
    where
        F: FnOnce(Vec<J>) -> Fut + Send + 'static,
        Fut: Future<Output = Vec<R>> + Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let member = Member {
            job,
            answer: tx,
            run: Box::new(move |jobs| Box::pin(run(jobs))),
            cap: cap.max(1),
        };
        let opened = {
            let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
            match queues.get_mut(&key) {
                Some(queue) => {
                    queue.push(member);
                    None
                }
                None => {
                    queues.insert(key.clone(), Vec::new());
                    Some(member)
                }
            }
        };
        if let Some(first) = opened {
            tokio::spawn(drive(self.queues.clone(), key, vec![first]));
        }
        rx.await.ok()
    }
}

/// Run `members` as one group, then every group the queue of `key` yields until it is
/// empty, and leave the key with no group running.
async fn drive<K, J, R>(queues: Queues<K, J, R>, key: K, mut members: Vec<Member<J, R>>)
where
    K: Eq + Hash,
{
    // Leaves the key with no group running however the groups end, a panic included: a
    // request that arrives after that starts its own group instead of queueing forever.
    let lease = Lease { queues, key };
    loop {
        let mut run = None;
        let (jobs, senders): (Vec<J>, Vec<oneshot::Sender<R>>) = members
            .into_iter()
            .map(|m| {
                run.get_or_insert(m.run);
                (m.job, m.answer)
            })
            .unzip();
        let Some(run) = run else {
            return;
        };
        let answers = run(jobs).await;
        for (tx, answer) in senders.into_iter().zip(answers) {
            // A request that went away no longer reads its answer.
            let _ = tx.send(answer);
        }
        members = {
            let mut queues = lease.queues.lock().unwrap_or_else(|e| e.into_inner());
            let Some(queue) = queues.get_mut(&lease.key) else {
                return;
            };
            if queue.is_empty() {
                return;
            }
            // The next group is the queue, up to the cap of the request that opens it;
            // the rest stay for the group after.
            let take = queue[0].cap.min(queue.len());
            queue.drain(..take).collect()
        };
    }
}

/// Removes a key's running group when its runner ends. A request still queued at that
/// point gets no answer; its oneshot is dropped with the queue.
struct Lease<K: Eq + Hash, J, R> {
    queues: Queues<K, J, R>,
    key: K,
}

impl<K: Eq + Hash, J, R> Drop for Lease<K, J, R> {
    fn drop(&mut self) {
        self.queues
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.key);
    }
}

/// How many rows of `per_row` bytes each a card takes beside one another: the free bytes
/// of the roomiest card divided by the row, at least one and at most `want`. With no card
/// the host holds them, and every row is taken.
pub fn rows_that_fit(per_row: u64, want: usize) -> usize {
    let most_free = crate::inference::place::room::card_rooms()
        .iter()
        .map(|c| c.available)
        .max();
    rows_in(most_free, per_row, want)
}

/// [`rows_that_fit`] given the roomiest card's free bytes, `None` when there is no card.
pub fn rows_in(most_free: Option<u64>, per_row: u64, want: usize) -> usize {
    let want = want.max(1);
    match most_free {
        Some(free) if per_row > 0 => {
            usize::try_from(free / per_row).map_or(want, |rows| rows.clamp(1, want))
        }
        _ => want,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Notify;

    /// What every group was given, one entry per call, in call order.
    type Calls = Arc<Mutex<Vec<Vec<u32>>>>;

    /// A group's answers, from a runner that waits before it answers.
    type Boxed<R> = Pin<Box<dyn Future<Output = Vec<R>> + Send>>;

    /// A runner that records its jobs and doubles each.
    fn doubling(calls: Calls) -> impl FnOnce(Vec<u32>) -> std::future::Ready<Vec<u32>> {
        move |jobs| {
            calls.lock().unwrap().push(jobs.clone());
            std::future::ready(jobs.into_iter().map(|j| j * 2).collect())
        }
    }

    /// A runner that records its jobs, waits for `release`, then doubles each: the group
    /// ahead, held open so the test can queue requests behind it.
    fn held(calls: Calls, release: Arc<Notify>) -> impl FnOnce(Vec<u32>) -> Boxed<u32> {
        move |jobs| {
            calls.lock().unwrap().push(jobs.clone());
            Box::pin(async move {
                release.notified().await;
                jobs.into_iter().map(|j| j * 2).collect()
            })
        }
    }

    /// Open a group of `first` on `gather` and keep it running until `release` is notified;
    /// returns the request's handle once its runner has started.
    async fn group_ahead(
        gather: &Gather<&'static str, u32, u32>,
        calls: &Calls,
        first: u32,
        release: &Arc<Notify>,
    ) -> tokio::task::JoinHandle<Option<u32>> {
        let gather = gather.clone();
        let runner = held(calls.clone(), release.clone());
        let ahead = tokio::spawn(async move { gather.submit("k", first, 4, runner).await });
        while calls.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        ahead
    }

    /// Submit `job` from a task of its own and yield until it has queued.
    async fn queued(
        gather: &Gather<&'static str, u32, u32>,
        calls: &Calls,
        job: u32,
        cap: usize,
    ) -> tokio::task::JoinHandle<Option<u32>> {
        let mine = gather.clone();
        let runner = doubling(calls.clone());
        let handle = tokio::spawn(async move { mine.submit("k", job, cap, runner).await });
        while gather
            .queues
            .lock()
            .unwrap()
            .get("k")
            .is_none_or(|q| !q.iter().any(|m| m.job == job))
        {
            tokio::task::yield_now().await;
        }
        handle
    }

    #[tokio::test]
    async fn a_lone_job_runs_without_waiting() {
        let gather: Gather<&str, u32, u32> = Gather::default();
        let calls = Calls::default();
        let mut request = Box::pin(gather.submit("k", 3, 4, doubling(calls.clone())));
        // The first poll registers the request and spawns its runner; one turn of the
        // scheduler is all the runner needs, since no timer stands between them.
        assert!(futures::poll!(request.as_mut()).is_pending());
        tokio::task::yield_now().await;
        assert_eq!(
            *calls.lock().unwrap(),
            [[3]],
            "the runner ran at the next turn"
        );
        assert_eq!(request.await, Some(6));
    }

    #[tokio::test]
    async fn jobs_arriving_during_a_run_form_the_next_group() {
        let gather: Gather<&str, u32, u32> = Gather::default();
        let calls = Calls::default();
        let release = Arc::new(Notify::new());
        let ahead = group_ahead(&gather, &calls, 1, &release).await;
        let second = queued(&gather, &calls, 2, 4).await;
        let third = queued(&gather, &calls, 3, 4).await;
        release.notify_one();
        assert_eq!(ahead.await.unwrap(), Some(2));
        assert_eq!(second.await.unwrap(), Some(4));
        assert_eq!(third.await.unwrap(), Some(6));
        assert_eq!(*calls.lock().unwrap(), [vec![1], vec![2, 3]]);
        assert!(
            gather.queues.lock().unwrap().is_empty(),
            "the key is free once its queue is drained"
        );
    }

    #[tokio::test]
    async fn requests_with_different_keys_run_apart() {
        let gather: Gather<&str, u32, u32> = Gather::default();
        let calls = Calls::default();
        let (a, b) = tokio::join!(
            gather.submit("one", 3, 4, doubling(calls.clone())),
            gather.submit("two", 5, 4, doubling(calls.clone())),
        );
        assert_eq!((a, b), (Some(6), Some(10)));
        assert_eq!(calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_group_holds_no_more_than_its_cap() {
        let gather: Gather<&str, u32, u32> = Gather::default();
        let calls = Calls::default();
        let release = Arc::new(Notify::new());
        let ahead = group_ahead(&gather, &calls, 1, &release).await;
        let waiting = [
            queued(&gather, &calls, 2, 2).await,
            queued(&gather, &calls, 3, 2).await,
            queued(&gather, &calls, 4, 2).await,
        ];
        release.notify_one();
        assert_eq!(ahead.await.unwrap(), Some(2));
        for (handle, want) in waiting.into_iter().zip([4, 6, 8]) {
            assert_eq!(handle.await.unwrap(), Some(want));
        }
        assert_eq!(
            *calls.lock().unwrap(),
            [vec![1], vec![2, 3], vec![4]],
            "two together, the third in the group after"
        );
    }

    #[tokio::test]
    async fn a_cap_of_one_never_groups() {
        let gather: Gather<&str, u32, u32> = Gather::default();
        let calls = Calls::default();
        let release = Arc::new(Notify::new());
        let ahead = group_ahead(&gather, &calls, 1, &release).await;
        let second = queued(&gather, &calls, 2, 1).await;
        let third = queued(&gather, &calls, 3, 1).await;
        release.notify_one();
        assert_eq!(ahead.await.unwrap(), Some(2));
        assert_eq!(second.await.unwrap(), Some(4));
        assert_eq!(third.await.unwrap(), Some(6));
        assert_eq!(*calls.lock().unwrap(), [vec![1], vec![2], vec![3]]);
    }

    #[tokio::test]
    async fn a_job_that_fails_in_a_group_keeps_its_companions_answers() {
        let gather: Gather<&str, u32, Result<u32, String>> = Gather::default();
        let calls = Arc::new(Mutex::new(0usize));
        let runner = |calls: Arc<Mutex<usize>>, release: Option<Arc<Notify>>| {
            move |jobs: Vec<u32>| {
                *calls.lock().unwrap() += 1;
                Box::pin(async move {
                    if let Some(release) = release {
                        release.notified().await;
                    }
                    jobs.into_iter()
                        .map(|j| {
                            if j == 0 {
                                Err("zero".to_string())
                            } else {
                                Ok(j * 2)
                            }
                        })
                        .collect::<Vec<_>>()
                }) as Boxed<Result<u32, String>>
            }
        };
        let release = Arc::new(Notify::new());
        let ahead = {
            let (gather, runner) = (gather.clone(), runner(calls.clone(), Some(release.clone())));
            tokio::spawn(async move { gather.submit("k", 1, 4, runner).await })
        };
        while *calls.lock().unwrap() == 0 {
            tokio::task::yield_now().await;
        }
        let mut behind = Vec::new();
        for job in [3, 0, 5] {
            let (mine, runner) = (gather.clone(), runner(calls.clone(), None));
            behind.push(tokio::spawn(async move {
                mine.submit("k", job, 4, runner).await
            }));
            while gather
                .queues
                .lock()
                .unwrap()
                .get("k")
                .is_none_or(|q| !q.iter().any(|m| m.job == job))
            {
                tokio::task::yield_now().await;
            }
        }
        release.notify_one();
        assert_eq!(ahead.await.unwrap(), Some(Ok(2)));
        let answers: Vec<_> = futures::future::join_all(behind)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(
            answers,
            [Some(Ok(6)), Some(Err("zero".to_string())), Some(Ok(10))]
        );
        assert_eq!(*calls.lock().unwrap(), 2, "the three ran as one group");
    }

    #[tokio::test]
    async fn a_runner_that_answers_short_leaves_the_rest_without_an_answer() {
        let gather: Gather<&str, u32, u32> = Gather::default();
        let calls = Calls::default();
        let release = Arc::new(Notify::new());
        let ahead = group_ahead(&gather, &calls, 1, &release).await;
        let short = |jobs: Vec<u32>| std::future::ready(vec![jobs[0]]);
        let second = {
            let gather = gather.clone();
            tokio::spawn(async move { gather.submit("k", 3, 4, short).await })
        };
        while gather
            .queues
            .lock()
            .unwrap()
            .get("k")
            .is_none_or(Vec::is_empty)
        {
            tokio::task::yield_now().await;
        }
        let third = {
            let gather = gather.clone();
            tokio::spawn(async move { gather.submit("k", 5, 4, short).await })
        };
        while gather
            .queues
            .lock()
            .unwrap()
            .get("k")
            .is_none_or(|q| q.len() < 2)
        {
            tokio::task::yield_now().await;
        }
        release.notify_one();
        assert_eq!(ahead.await.unwrap(), Some(2));
        assert_eq!(second.await.unwrap(), Some(3));
        assert_eq!(third.await.unwrap(), None);
    }

    #[test]
    fn rows_follow_the_free_bytes_of_the_roomiest_card() {
        assert_eq!(rows_in(None, 10, 4), 4, "the host takes every row");
        assert_eq!(rows_in(Some(25), 10, 4), 2);
        assert_eq!(rows_in(Some(5), 10, 4), 1, "one row always runs");
        assert_eq!(rows_in(Some(1000), 10, 4), 4);
        assert_eq!(rows_in(Some(1000), 0, 3), 3);
    }
}
