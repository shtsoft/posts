# Faking Algebraic Effects and Handlers With Traits: A Rust Design Pattern

Algebraic effects and handlers have become a hot topic in programming language research over the last two decades.
In the early days of handlers, people were curious but often found it difficult to get started with the topic.
As a result, one of the pioneers of handlers eventually felt compelled to write the tutorial [An Introduction to Algebraic Effects and Handlers](http://www.sciencedirect.com/science/article/pii/S1571066115000705) to make the subject more accessible.\
The tutorial begins with: "*Algebraic effects* are an approach to computational effects based on the premise that impure behavior arises from a set of *operations*."
Rust provides a way to work with "sets of operations": traits.
But does that mean we can actually implement algebraic effects and handlers in vanilla Rust?
As it turns out, not really.
However, attempting to fake them leads to a potentially interesting Rust design pattern.

This post explores that idea.
More precisely, after a brief discussion of effect handlers and a Rust encoding of them ([The Try](#the-try)), we examine the shortcomings of this encoding using a canonical effect handler example ([The Trickery](#the-trickery)).
In the subsequent discussion, however, more sophisticated examples involving async/await and probabilistic programming suggest that the encoding may still be useful ([The Usefulness](#the-usefulness)).\
That said, you should not expect a revolution.
The pattern is very much down to earth and is probably already in use somewhere.

## The Try

This section explains effect handlers and a possible Rust encoding in more detail.
In particular, examining the idea of effect handlers will allow us to derive an encoding in Rust.
We will also discuss some general properties of the resulting pattern.

To properly understand the relationship between effect handlers and Rust traits, we first need a more concrete definition of effects and handlers.
In general, an effect is something that influences its environment or context.
If effects are modeled as sets of operations, then these operations must be able to make use of the context in which they are executed.
Handling such an effect therefore means specifying what the corresponding operations do depending on their context.
A handler is simply an implementation of those operations where the implementation has access to the context.
However, this access must remain abstract, because the implementation cannot know in advance what the actual calling context of the operations will be.

Having established what we mean by effects and handlers, let us look at how to model this idea in Rust.
While it is clear what the "sets of operations" are - namely, ordinary traits - it is less obvious how to make the context accessible to the trait methods.
To address this problem, first note that making the context accessible effectively means making the continuation accessible.
However, vanilla Rust does not support (delimited) continuations.
Therefore, some workaround is required.
The usual way to obtain access to the continuation is to use continuation-passing style (CPS).
Indeed, the syntax of the language in [An Introduction to Algebraic Effects and Handlers](http://www.sciencedirect.com/science/article/pii/S1571066115000705) suggests writing the trait methods in CPS.
However, the semantics enforced by Rust when doing so do not exactly correspond to the intended semantics described in that article.
Still, it appears to be the only low-effort workaround for making the context accessible.
So, without further ado, here is what the general pattern for an effect with a single operation looks like:

```rust
trait Effect<A, B, C> {
    fn operation<K>(self, v: A, k: K) -> C
    where
        K: FnOnce(B) -> C;
}
```

A handler, then, is simply an implementation of the trait.
In general, the idea is to treat CPS-style interfaces as effects and their implementations as handlers.

Before examining the applicability of this idea, we can already make some general remarks about the pattern.\
First, note that it inherits the advantages typically associated with Rust: safety, low-level applicability, and efficiency.
In particular, efficiency here means that code parameterized over handlers is specialized to concrete handlers via monomorphization.\
However, it also inherits some of Rust’s peculiarities.
Rust’s affine type system prevents arbitrary use of the continuation `k` if it is bounded by `FnOnce`.
Therefore, one has to choose the trait bound for `K` carefully.
To be clear, we view this less as a limitation and more as a refinement.\
Another notable aspect is that the handlers are *shallow* as a consequence of being trait implementations.
That is, a handler is consumed by an operation call and must be explicitly cloned (and possibly passed along) if it is needed again elsewhere.
This contrasts with the *deep* handlers commonly found in languages that natively support effect handlers.

## The Trickery

This section discusses the viability of the pattern.
The discussion is based on a standard example of effect handlers: exception handlers.\
[Exception handling](#exception-handling-code) is probably the most widely known example of effect handling.
Discussing it will hopefully help you better understand the pattern, and if you are new to the topic, effect handlers in general.
However, it will also reveal a major shortcoming of the low-level approach of interpreting effects using explicit CPS.

### Exception Handling ([Code](https://gist.github.com/shtsoft/81520bb6b1c35e1bf497088438cd8cea))

If you are a programmer who has worked with multiple languages, you have likely encountered exception handlers before.
In fact, effect handlers can be seen as a generalization of exception handlers.
The corresponding exception effect can generally be defined as follows:

```rust
trait Exception<B> {
    fn raise<A, K>(self, v: (), k: K) -> B
    where
        K: FnOnce(A) -> B;
}
```

The generic `A` essentially ensures that the continuation `k` cannot be used in the handler implementation, which is the defining property of an exception.\
Now, any type implementing this trait acts as an exception handler.
For example, handling division by zero could be implemented by simply printing an error message and discarding the continuation:

```rust
struct ExceptionDiv {}

impl Exception<()> for ExceptionDiv {
    fn raise<A, K>(self, _: (), _: K)
    where
        K: FnOnce(A),
    {
        println!("Error: Division By Zero.");
    }
}
```

Then, actually raising an exception in a function implementing integer division could look like this:

```rust
fn div(a: usize, b: usize, exception_handler: impl Exception<()>) {
    let continuation = |x| {
        println!("Result: {x}");
    };

    if b == 0 {
        exception_handler.raise((), continuation);
    } else {
        continuation(a / b);
    }
}

div(5, 2, ExceptionDiv {}); // prints `2`
div(5, 0, ExceptionDiv {}); // prints the error message defined above by `ExceptionDiv`
```

What is interesting here is how this pattern compares to ordinary exception/effect handler semantics.
In a language with exception handlers, one would write the above function body more like this:

```python
try:
    let x =
        if b == 0:
            raise
        else:
            a / b
    print("Result: {x}")
except:
   exception_handler
```

having the language dynamically derive the continuation.
The same would also be true in a language with effect handlers.
However, in the Rust approach one has to push the continuation into the branches manually.
This is problematic.
If anything, a compiler should perform such transformations automatically.
For human programmers, the severe negative impact on ergonomics is hardly acceptable.\
And it only gets worse.
Other control-flow constructs suffer from similar issues.
For example, the problem with `if` generalizes to other branching constructs such as `match`.
Furthermore, effectful `loop`s must be replaced by recursive functions, which introduces the serious risk of stack overflows.\
In general, the problem lies in implicit *jumps* within the continuation.
A workaround is to make these jumps explicit.
This is essentially what “pushing into the branches” and “replacing loops with recursive functions” accomplish.\
The implicit jumps in branching and looping are local in the sense that the jumps remain within a single function body.
But what if the context of an effect crosses function boundaries?
These non-local jumps must also be made explicit.
A generic way to achieve this is to write the function containing such an effect operation in CPS.
In other words, to achieve true modularity, effectful functions must themselves be written in CPS.\
For example, the `div` function from above becomes:

```rust
fn try_div<C>(a: usize, b: usize, exception_handler: impl Exception<()>, k: impl Fn() -> C) -> C {
    let continuation = |x| {
        println!("Result: {x}");
        k()
    };

    if b == 0 {
        exception_handler.raise((), continuation);
    } else {
        continuation(a / b);
    }
}
```

This has to be contrasted with a language that supports exception or effect handlers.
In such a language, the code would look more like this:

```python
div(a: int, b: int):
    ...
        raise
    ...

try:
    div(...)
    k()
except:
   exception_handler
```

In a language with effect handlers, it would look very similar.
Just replace `except` with something like `with`.
This approach is clearer, less verbose, and - most importantly - not burdened by CPS.
Indeed, one of the major promises of effect handlers is to abstract and modularize control flow *without* CPS, while remaining clear and concise.
For that reason, the low-level approach to effect handling presented here is essentially a fake.
It goes against the original spirit of effect handlers and cannot really compete with them as a language construct.\
Yet, the idea should not be discarded too quickly.
While it is true that a general solution to the problem of effectful functions requires CPS, this does not necessarily mean that the programmer must write the entire program in CPS - perhaps only small parts of it.
It is also conceivable that, in some situations, the issues can be circumvented elegantly in a problem-specific way.
Moreover, it is not always clear whether these issues will arise in practice or even need to be considered problematic.
To what extent the pattern can still help abstract and modularize control is the topic of the next section.

## The Usefulness

This section suggests that the pattern could, in fact, be useful for abstracting and modularizing control.
To explore this idea, the discussion revolves around two examples that are traditionally used to showcase effect handlers:

- an abstraction of [async/await](#asyncawait-code)
- a modularization of statistical models and statistical inference enabling [probabilistic programming](#probabilistic-programming-code)

The discussion does not attempt to rigorously assess the usefulness of the pattern based on these examples; instead, that task is left to the reader.

Before moving on to these more practical examples, however, we briefly discuss [shift/reset](#shiftreset-code).

This is not necessarily because of its usefulness, but rather because it is somewhat better known in mainstream programming circles (compared to effect handlers).
As such, it may help the reader better grasp the power of the pattern beyond viewing effect operations merely as a somewhat unusual encoding of higher-order functions.

### Shift/Reset ([Code](https://gist.github.com/shtsoft/fbec95185b928713607e1883e5669e81))

The idea behind `shift`/`reset` is to reify the portion of the continuation enclosed by the two keywords.
In terms of expressive power, it is equivalent to effect handlers and can therefore be modeled using them.
Consequently, it is not surprising that the pattern presented here can also produce a rough imitation of it.
One possible approach is to treat `reset` as the effect operation that expects a `shift`, which binds the continuation within a `reset` computation:

```rust
trait Reset<C> {
    fn reset<B, K>(self, shift: impl FnOnce(K) -> C, k: K) -> C
    where
        K: FnOnce(B) -> C;
}
```

The handler providing the usual shift/reset semantics for `reset` just calls back the `shift` on the actual continuation `k`:[^1]

```rust
struct CallBack {}
impl<C> Reset<C> for CallBack {
    fn reset<B, K>(self, shift: impl FnOnce(K) -> C, k: K) -> C
    where
        K: FnOnce(B) -> C,
    {
        shift(k)
    }
}
```
Note that `shift` itself looks like an effect operation and also note how it is trivially `handled` in `CallBack`.\
A silly example (where the continuation is used twice) is:

```rust
fn thirteen(reset_handler: impl Reset<usize>) {
    let shift_k = |k: fn(usize) -> usize| k(1) + k(3);
    let thirteen = 5 + reset_handler.reset(shift_k, |n| n + 2);

    println!("Result: {thirteen}",);
}

thirteen(CallBack {});
```

Since the handler for `reset` is usually provided as `CallBack {}`, it can be convenient to hide `CallBack {}.reset(...)` behind a macro `reset!()` and instead write:

```rust
let thirteen = 5 + reset!(|k: fn(usize) -> usize| k(1) + k(3), |n| n + 2);

println!("Result: {thirteen}",);
```

This approach reduces the boilerplate that arises from the handlers being shallow in this context.

### Async/Await ([Code](https://gist.github.com/shtsoft/829e2f161fda0dd0892b521febe6624b))

The concept of async/await is to execute two jobs asynchronously, where the second job may await the result of the first at certain points during its execution.
This idea involves two effect operations: one for executing jobs asynchronously and another for awaiting a result.
This control structure can be abstracted using the following effect interface:

```rust
trait AsyncAwait<B, C> {
    type Promise<X>;
    fn asynchronous<K>(self, asynchronous_b: fn() -> B, k: K) -> C
    where
        K: FnOnce(Self::Promise<B>) -> C,
        Self: Sized;
    fn awaiting<K>(self, promise_b: Self::Promise<B>, k: K) -> C
    where
        K: FnOnce(B) -> C;
}
```

Here, the thunk `asynchronous_b` represents the first job, and the continuation of `asynchronous`, parameterized over the promised result, represents the second job, which eventually depends on the first.
In particular, the second job may effectfully `await` the result of the first job (`promise_b`).\
Thus, handlers need to somehow pipe the output of the `asynchronous` operation to the input of the `awaiting` operation while executing both jobs asynchronously.
A very straightforward Rust implementation of this behavior is to link them using a `channel()`.
Doing so results in:

```rust
struct Channel {}
impl<B: Send + 'static, C> AsyncAwait<B, C> for Channel {
    type Promise<X> = Receiver<X>;
    fn asynchronous<K>(self, asynchronous_b: fn() -> B, k: K) -> C
    where
        K: FnOnce(Self::Promise<B>) -> C,
    {
        let (tx, rx) = channel();

        thread::spawn(move || {
            let promise_b = asynchronous_b();
            tx.send(promise_b).unwrap();
        });

        k(rx)
    }
    fn awaiting<K>(self, promise_b: Self::Promise<B>, k: K) -> C
    where
        K: FnOnce(B) -> C,
    {
        let b = promise_b.recv().unwrap();
        k(b)
    }
}
```

Note that a new thread is spawned each time `asynchronous` is called.
This is costly, and a real-world implementation would instead use a thread pool.
Otherwise, however, this approach behaves largely as one would expect from an async/await implementation (w.r.t. to our definition but not w.r.t. the fork/yield-like definition used in real-world languages).\
A major advantage of async/await as an abstract control structure is the modularity it provides: a handler can be implemented anywhere by anyone, and code can abstract over the handler, making it widely reusable.
In this sense, the handler abstractions in the code can be understood as requirements on the caller to provide the respective capabilities - similar to requiring an async/await capability.[^2]
For example, downloading a resource and returning its size while performing other tasks until the size is needed could look like this:

```rust
fn download_doing_stuff(async_handler: impl AsyncAwait<usize, ()> + Copy) {
    fn download() -> usize {
        ...
    }

    fn do_stuff() {
        ...
    }

    fn do_other_stuff(size: usize) {
        ...
    }

    async_handler.asynchronous(download, |size| {
        do_stuff();
        async_handler.awaiting(size, do_other_stuff);
    });
}
```

This function can be called with the handler defined above:

```rust
download_doing_stuff(Channel {});
```

But it could also be called with a different handler.
For instance, on a platform without support for spawning threads, `download_doing_stuff` does not need to be modified; it only needs to be invoked with an alternative handler.
`download_doing_stuff` simply requires an async/await capability.\
That said, in the case of async/await, a global handler is likely the appropriate choice most of the time.
Using the respective macros instead of a handler abstraction, the code might look like this:

```rust
async!(download, |size| {
    do_stuff;
    await!(size, do_other_stuff)
});
```

This actually looks nice, doesn’t it?\
One caveat, however: the CPS involved in the effect interface makes this pattern probably unsuitable for massively asynchronous code with complex concurrency structures.
The control abstraction presented here should be considered a lightweight solution for programs that frequently use isolated asynchronous jobs.
Nevertheless, it can be quite effective when there is a need to work with different async/await implementations.

### Probabilistic Programming ([Code](https://gist.github.com/shtsoft/ba971c25e5bd559cc19cd07dff9dc761))

The idea of probabilistic programming is to automate “(statistical) inference” on “(statistical) models.”\
More precisely, a model is a formal description of some (dynamic) aspect of reality, an idea, or whatever, which produces an outcome when executed.
If the description involves randomness, the model is considered statistical.
Since the outcome depends on random choices, the outcome itself is not particularly interesting.
What matters is inferring the distribution of outcomes by repeatedly executing the model and accounting for the likelihood of each outcome relative to the random choices made.\
Probabilistic programming aims to encapsulate this inference in library code.\
However, there is a modularity problem: the likelihood is implicitly baked into within the model, but it is needed externally for inference.
A modern solution is to treat recording the likelihood as an effect of the model, which is handled by the inference mechanism and thus accessible as state information at the recording points.
In Rust, this could be expressed as:

```rust
struct Particle<Handler, Outcome> {
    pub handler: Handler,
    pub outcome: Option<Outcome>,
}

type Model<H, O> = fn(H) -> Particle<H, O>;
type Continuation<'a, H, O> = Box<dyn Fn(H) -> Particle<H, O> + 'a>;

trait Likelihood<'a, O: Copy + 'a> {
    fn score(self, p: f64, k: Continuation<'a, Self, O>) -> Particle<Self, O>
    where
        Self: Sized;
}
```

`score` is the effect operation for recording a portion of the likelihood, `p`.
The operation returns the handler instead of just `()` because the handler is shallow and serves as a parameter of the model intended to carry state information accessible to the inference algorithm.
Consequently, the handler must be threaded through the model, and in particular through the continuation.
This leads to types for `Model` and `Continuation` analogous to a state monad parameterized over the outcome, with the continuation stored in a `Box` to allow it to be part of the state.\
Also note that the name `Particle` for the result type of `Model` is common in statistical modeling.
It appears to originate from physics, representing a classical particle.
Interestingly, it represents the outcome in the context of the handler state, making it a “statistical” particle that slightly resembles a quantum particle.\
Now, handlers must record the likelihood according to the needs of the inference algorithm.
For importance sampling - a method that approximates the posterior distribution from Bayesian inference using Bayes’ theorem - it is sufficient to accumulate the likelihood contributions.
One way to achieve this is to wrap the weight accumulator for the likelihood inside the handler that carries its state:

```rust
struct WeighWeight {
    weight: f64,
}
impl WeighWeight {
    const fn new(weight: f64) -> Self {
        Self { weight }
    }
}
impl<'a, O: Copy + 'a> Likelihood<'a, O> for WeighWeight {
    fn score(self, p: f64, k: Continuation<'a, Self, O>) -> Particle<Self, O> {
        k(Self::new(self.weight * p))
    }
}
```

An implementation of importance sampling based on the `WeighWeight` handler is:

```rust
type Posterior<Outcome> = Vec<(f64, Outcome)>;

const DEFAULT_WEIGHT: f64 = 1.0;

fn importance_sampling<'a, O: Copy + 'a>(
    model: Model<WeighWeight, O>,
    number_of_particles: usize,
) -> Posterior<O> {
    let mut particles = Vec::with_capacity(number_of_particles);

    // execute the model many times
    for _ in 0..number_of_particles {
        particles.push(model(WeighWeight::new(DEFAULT_WEIGHT)));
    }

    // measure the current weight of the particles
    particles
        .iter()
        .map(|particle| (particle.handler.weight, particle.outcome.unwrap()))
        .collect()
}
```

Eventually, one will also need functions to visualize posteriors; e.g.:

```rust
fn print_bool_distribution(posterior: Posterior<bool>) {
    let mut trues = 0.0;
    let mut falses = 0.0;

    for (weight, outcome) in posterior {
        if outcome {
            trues += weight;
        } else {
            falses += weight;
        }
    }

    println!(
        "BOOL DISTRIBUTION: {:.2}% TRUE vs {:.2}% FALSE",
        trues * 100.0 / (trues + falses),
        falses * 100.0 / (trues + falses)
    );
}
```

One of the benefits of probabilistic programming is that the implementation details above are usually irrelevant to most probabilistic programmers.
Typically, they only need to implement their model and select an inference algorithm and visualization tool from a library.\
A common toy model used to illustrate probabilistic programming is the so-called *sprinkler model*.
It statistically describes how rain and a sprinkler, treated as random events, influence whether a lawn is wet, ultimately producing an output indicating if it is raining.

```rust
use rand::distributions::{Bernoulli, Distribution};
use rand::thread_rng;

fn sprinkler<'a, H>(score_handler: H) -> Particle<H, bool>
where
    H: Likelihood<'a, bool>,
{
    let rain = Bernoulli::new(0.2).unwrap().sample(&mut thread_rng());
    let sprinkler = Bernoulli::new(0.1).unwrap().sample(&mut thread_rng());

    let probability_lawn_wet = if rain {
        if sprinkler {
            0.99
        } else {
            0.7
        }
    } else if sprinkler {
        0.9
    } else {
        0.01
    };

    score_handler.score(
        probability_lawn_wet,
        Box::new(move |score_handler| Particle {
            handler: score_handler,
            outcome: Some(rain),
        }),
    )
}
```

It can be used to infer the probability of rain under the condition that the lawn is wet:

```rust
print_bool_distribution(importance_sampling(sprinkler, 1000));
```

The resulting posterior is something like `65% TRUE vs. 35% FALSE`.\
This outcome could also have been obtained using another inference algorithm, such as sequential Monte Carlo (SMC).
Having the choice of algorithm can be particularly important for efficiency.
For example, importance sampling may sometimes require significantly more particles than SMC to achieve equally accurate approximations.\
On the other hand, importance sampling could be applied to a different model, such as linear regression.
An implementation of that (and of SMC) can be found [here](https://gist.github.com/shtsoft/ba971c25e5bd559cc19cd07dff9dc761).\
This demonstrates that non-trivial probabilistic programming is possible with this pattern.
Importantly, the [code](https://gist.github.com/shtsoft/ba971c25e5bd559cc19cd07dff9dc761) remains reasonably readable and performs well.
Finally, the applicability of the pattern in this more complex scenario provides confidence in its usefulness.

[^1]: Actually, it is more a matter of control/prompt due to the shallowness of the handlers here, but we will ignore that.
[^2]: The [Effekt language](https://effekt-lang.org) generally treats handlers as capabilities. In fact, its semantics is defined via an interpretation in a language enforcing explicit capability-passing style.
