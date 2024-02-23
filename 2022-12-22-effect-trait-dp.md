---
heading: "Faking Algebraic Effects and Handlers With Traits: A Rust Design Pattern"
---

[]()
Algebraic effects and handlers have become a hot topic in programming language research during the last two decades.
In the early days of handlers people were curious but seemed to have difficulties to get started with the topic.
And so one of the pioneers of handlers eventually felt compelled to write the tutorial [An Introduction to Algebraic Effects and Handlers](http://www.sciencedirect.com/science/article/pii/S1571066115000705) to make it more accessible.\
The tutorial starts with '*Algebraic effects* are an approach to computational effects based on a premise that impure behaviour arises from a set of *operations*'.
Rust does feature a way to deal with 'sets of operations': traits.
But does that also mean one can actually do algebraic effects and handlers with vanilla Rust?
As it turns out, not really.
But trying to fake it leads to a perhaps interesting Rust design pattern.

This is what this post is about.
More precisely, after a brief discussion of effect handlers and an encoding of it in Rust ([The Try](#the-try)), the shortcomings of the encoding are addressed based on an archetypical effect handler example ([The Trickery](#the-trickery)).
However, in the subsequent discussion more sophisticated examples regarding async/await and probabilisitc programming will suggest that the encoding could still be useful ([The Usefulness](#the-usefulness)).
Yet, you should not expect a revolution.
The pattern is very much down to earth and probably already deployed somewhere.

## The Try

This section explains effect handlers and a possible Rust encoding more thoroughly.
More precisely, contemplating the idea of effect handlers will allow us to encode effect handlers in Rust.
Moreover, we address some general properties of the resulting pattern.

To properly understand the relation between effect handlers and Rust traits we first need a more concrete definition of effects and handlers.
Generally, an effect is something which has an impact on its environment/context.
So, if effects are modeled as sets of operations, then these operations must have the ability to make use of the context they are executed in.
Handling such an effect then means to specify what the according operations do depending on their context.
So a handler is just an implementation of the according operations where the implementation has the ability to make use of the context.
However, the use of the context has to be abstract, because the implementation cannot know in advance what the actual calling context of the operations are.\
Having established what we consider effects and handlers, let us look at how to model the idea with Rust.
While it is clear what the 'sets of operations' are - namely, just ordinary traits - it is not so clear how to make the context accessible to the traits' methods.
To address that problem, first note that making the context accessible means to make the continuation accessible here.
But vanilla Rust lacks support for (delimited) continuations.
Therefore some workaround is required.
The usual way to get a hold of the continuation is to use continuation-passing style (CPS).
In fact, the syntax of the language from [An Introduction to Algebraic Effects and Handlers](http://www.sciencedirect.com/science/article/pii/S1571066115000705) suggests to write the methods of the trait in CPS.
However, the semantics enforced by Rust in doing so does not quite correspond to the intended semantics of that article.
Still, it seems to be the only low-effort workaround to make the context accessible.
So without further ado here is what the general pattern for an effect with one operation looks like:

```rust
trait Effect<A, B, C> {
    fn operation<K>(self, v: A, k: K) -> C
    where
        K: FnOnce(B) -> C;
}
```

A handler then simply is an implementation of the trait.
Generally, the idea is to consider interfaces in CPS effects and implementations thereof handlers.

Before examining the applicability of that idea we can already make some general remarks on the pattern.\
First, note that the pattern shares the advantages of Rust-like saftey, low-level applicability and efficiency.
In particular, efficiency means that code parameterized over handlers is specialized to the handlers by monomorphization.
But it also shares its peculiarities.
Rust's affine type systems prevents an arbitrary use of the continuation `k` if it is bounded by `FnOnce`.
So one has to choose the trait bound for `K` wisely.
To be clear, we regard that less as limitation but more as refinement.\
Otherwise, it is notable that the handlers are 'shallow' as a consequence of being a trait implementation.
That is, a handler is consumed by an operation call and has to be explcitly cloned (and possibly passed) if needed again somewhere else.
This is in contrast to the 'deep' handlers commonly found in languages with effect handlers.

## The Trickery

This section discusses the viability of the pattern.
The discussion is based on a standard example of effect handlers: exception handlers.
[Exception handling](#exception-handling-code) is the probably most widely known example of effect handling.
Discussing it will hopefully help you to understand the pattern better and if you are new to the topic, then also effect handlers in general.
Yet, it will also reveal a big shortcoming of the low-level approach of interpreting effects in explicit CPS.

### Exception Handling ([Code](https://gist.github.com/shtsoft/81520bb6b1c35e1bf497088438cd8cea))

If you are a programmer who has worked with many languages, you might have already come into touch with exception handlers.
Either way, effect handlers generalize exception handlers.
The according exception effect can generally be defined as follows:

```rust
trait Exception<B> {
    fn raise<A, K>(self, v: (), k: K) -> B
    where
        K: FnOnce(A) -> B;
}
```

The generic `A` essentially enforces that the continuation `k` cannot be used in the handler implementation which is the defining property of an exception.\
Now, everything implementing that trait is an exception handler.
For example, handling divisions by zero could be implemented by just printing an error message and throwing away the continuation:

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

Then, actually raising an exception in a function implementing integer division could look like that:

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

What is interesting here is how the pattern compares to the ordinary exception/effect handler semantics.
In a language with exception handlers one would write the above function body rather as

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
And that's also what it would be like in a language with effect handlers.
However, in the Rust approach one has to push the continuation into the branches manually.
This is bad.
If at all, a compiler should do that.
For human programmers that severe negative impact on the ergonomics is hardly acceptable.\
And it only gets worse.
Other control flow constructs suffer from similar problems.
For example, the problem with `if` generalizes to other branching constructs like `match`.
Furthermore, effectful `loop`s have to be replaced by recursive functions which has the serious drawback of perhaps overflowing the stack.\
Generally, the problem is implicit 'jumping' within the continuation.
A workaround is making that explicit.
This is what 'pushing into the branches' and 'replacing by recursive function' essentially do.\
The implicit jumps in branching and looping are local in the sense that jumping is within a single function body.
But what if the context of an effect crosses function borders?
These non-local jumps have to be made explicit, too.
A generic way to do so is to write the function containing such an effect operation in CPS.
That is, to be really modular one has to write effectful functions in CPS.\
For example, `div` from above becomes:

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

This has to be contrasted with a language with exception/effect handlers.
In a language with exception handlers the code would rather look like that:

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

In a language with effect handlers it would be very similar.
Just replace `except` with something like `with`.
This is clearer, less verbose and most importantly not accompanied by the burden of CPS.
Indeed, the major promise of effect handlers is to abstarct and modularize control without CPS in a clear and concise way.
So the low-level approach to effect handling presented here is a fake.
It is against the original spirit of effect handlers and can't really compete with effect handlers as language construct.\
Yet, one should not rashly scrap the idea.
While it is true that a general solution to the effectful functions problem requires CPS, it does not enforce the programmer to write the whole program in CPS but maybe just small portions of it.
It is also conceivable that it is sometimes possible to elegantly circumvent the issues in a problem-specific manner.
Moreover, it is not clear if the issues always come into play or have to be considered issues.
In how far the pattern can still abstract and modularize control is the topic of the next section.

## The Usefulness

This section suggests that the pattern could, in fact, be useful for abstracting and modularizing control.
To this end, the discussion revolves around two examples having a tradition in advertising effect handlers:
- an abstraction of [async/await](#asyncawait-code) is outlined
- a modularization of statistical model and statistical inference enabling [probabilistic programming](#probabilistic-programming-code) is outlined

The discussion itself does not make great efforts to assess the usefulness on the basis of those examples but leaves that task to reader.

But before begining with those more real-world examples, we briefly discuss [shift/reset](#shiftreset-code).
Not necessarily because of it usefulness but rather due to its better mainstream fame (compared to effect handlers) helping the reader to better grasp the power of the pattern beyond effect operations as somewhat strange encoding of higher-order functions.

### Shift/Reset ([Code](https://gist.github.com/shtsoft/fbec95185b928713607e1883e5669e81))

The idea of shift/reset is to reify the part of the continuation enclosed by the two keywords.
It is equivalent in power to effect handlers and hence can be modeled by them.
So it is no surprise that the pattern here can model a fake of it, at least.
One possibility is to make `reset` the effect operation expecting a `shift` which binds the continuation in some `reset`-computation:

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
Note that `shift` itself looks like an effect operation and how it is trivially `handled` in `CallBack`.\
A silly example (where the continuation is used twice) is:

```rust
fn thirteen(reset_handler: impl Reset<usize>) {
    let shift_k = |k: fn(usize) -> usize| k(1) + k(3);
    let thirteen = 5 + reset_handler.reset(shift_k, |n| n + 2);

    println!("Result: {thirteen}",);
}

thirteen(CallBack {});
```

As the handler of `reset` is usually provided as `CallBack {}` it can make sense to hide `CallBack {}.reset(...)` behind a macro `reset!()` and rather write:

```rust
let thirteen = 5 + reset!(|k: fn(usize) -> usize| k(1) + k(3), |n| n + 2);

println!("Result: {thirteen}",);
```

It mitigates the boilerplate emerging from the handlers being shallow here.

### Async/Await ([Code](https://gist.github.com/shtsoft/829e2f161fda0dd0892b521febe6624b))

The idea of async/await is to asynchronously do two jobs where the second perhaps awaits the result of the first at some points of its execution.
The idea contains two effect operations.
One for asynchronously doing the jobs and another for awaiting a result.
This control structure can be abstracted by the following effect interface:

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

Here, the thunk `asynchronous_b` is intended to be the first job and the continuation of `asynchronous` parameterized over the promised result is intended to be the second one, eventually manifesting the dependency on the first.
In particular, the second job is perhaps effectfully `awaiting` the result of the first one (`promise_b`).\
So, handlers ought to somehow pipe the output of the `asynchronous`-operation to the input of the `awaiting`-operation while asynchronously running the two jobs.
A very straightforward Rust implementation with this behavior is to link them by a `channel()`.
Doing so yields:

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

Note that a new thread is spawned each time `asynchrounous` is called.
This is costly and a real-world implementation would not do that but use a thread-pool instead, of course.
However, otherwise it is pretty much what one expects of an async/await-implementation.\
A major advantage of async/await as abstract control structure is the modularity coming along with it: a handler can be implemented everywhere by everybody and code can abstract over the handler making it more widely applicable.
In that regard, the handler abstractions in code can be understood as requirements on the caller to provide the respective capabilities - like the requirement of an async/await-capability.[^2]
For example, downloading something and returning its size while doing stuff until the size is needed to do other stuff can look like:

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

But it could also be called with a different one.
For instance, on a platform without support for spawning a thread, `download_doing_stuff` does not have to be changed but only be called with an alternative handler.
`download_doing_stuff` plainly requires an async/await-capability.\
Yet, in the case of async/await a global handler is likely the right choice most of the time.
And using the respective macros instead of a handler abstraction makes look the code like that:

```rust
async!(download, |size| {
    do_stuff;
    await!(size, do_other_stuff)
});
```

This actually looks nice, doesn't it?\
One warning has to be issued though.
The CPS involved in the effect-interface makes the pattern probably a bad choice for massively asynchronous code with a complicated concurrency-structure.
The control abstraction presented here should rather be considered a lightweight solution for programs tending to often use isolated asynchronous jobs.
But it can shine if there is a need to work on different async/await implementations.

### Probabilistic Programming ([Code](https://gist.github.com/shtsoft/ba971c25e5bd559cc19cd07dff9dc761))

The idea of probabilistic programming is to automize '(statistical) inference' on '(statistical) models'.\
More precisely, a model here is a description of some (dynamic) reality or idea or whatever in a formal language with an outcome when executed.
If the description involves some kind of randomness, the model is said to be statistical.
As the outcome depends on the randomness, it is not particularly interesting in itself.
What is interesting, however, is inferring the distribution of outcomes by executing the model again and again and using the respective likelihoods of the outcomes w.r.t. the random choices made in the model.
The idea of probabilistic programming is to turn that inference into library code.\
But there is a problem with modularity: the likelihood is implicitly baked into the model but rather needed outside of it for inference.
A modern solution to that problem is to consider the recording of the likelihood an effect of the model which is handled by the inference and hence accessible there as state information at the recording points.
In Rust this could look like:

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

`score` is the effect operation of recording a part of the likelihood, `p`.
The operation returns the handler instead of only a unit.
This is because the handler is shallow and is a parameter of the model intended to carry state information accessible to the inference algorithms.
So it has to be threaded through the model and in particular through the continuation.
This results in types for `Model` and `Continuation` analogous to the state monad parameterized over the outcome, where the continuation is in a `Box` to allow for making it part of the state.\
Also note that the name `Particle` for the result type of `Model` is common for a run of a statistical model.
It seems to have its origin in physics representing a classical particle there.
Interestingly, it can be considered the outcome in the context of the handler state, making it a 'statistical' particle slightly resembling a quantum particle.\
Now handlers have to somehow record the likelihood according to the inference algorithm's needs.
For importance sampling - an inference algorithm approximating the posterior distribution from Bayesian inference according to Bayes' theorem - it suffices to accumulate the parts of the likelihood.
One possibility to do that is to wrap the weight accumulator for the likelihood into the handler carrying its state:

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

Now, the perk of probabilistic programming is that the implementation of the above doesn't matter to most probabilistic programmers.
Usually, they only have to implement their model and choose an inference algorithm and visualization algorithm from a library.\
A typical toy model to illustrate probabilistic programming is the so-called sprinkler model.
It describes the influence of rain and a sprinkler as random events on wet lawn statistically, outputting if it is raining.

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
This result could have been obtained by some other inference alogrithm like sequential Monte Carlo (SMC), too.
Having the choice can be particularly important with respect to efficiency.
For example, importance sampling may sometimes need significantly more particles than SMC for equally good approximations.\
On the other hand, importance sampling could have been applied to some other model like linear regression.
An implementation of that (and of SMC) can be found [here](https://gist.github.com/shtsoft/ba971c25e5bd559cc19cd07dff9dc761).\
So, it is possible to do some non-trivial probabilistic programming with the pattern.
And, importantly, the [code](https://gist.github.com/shtsoft/ba971c25e5bd559cc19cd07dff9dc761) does not look too bad and performs decently.
Finally, we think the applicability in this more complex situation gives some confidence in the usefulness of the pattern.

[^1]: Actually, it is rather control/prompt due the shallowness of the handlers here. But we shall ignore that.
[^2]: The [Effekt language](https://effekt-lang.org) generally thinks of handlers as capabilities. In fact, its semantics is determined by an interpretation in a language enforcing explicit capability-passing style.
