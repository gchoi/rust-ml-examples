# candle-falcon

Falcon is a general large language model.

## Running an example

Make sure to include the `--use-f32` flag if using CPU, because there isn't a BFloat16 implementation yet.

```bash
# CPU
$ cargo run --release -- \
  --prompt "Flying monkeys are"

# CUDA
$ cargo run --features cuda --release -- \
  --prompt "Flying monkeys are" \
  --use-f32

# Metal
$ cargo run --features metal --release -- \
  --prompt "Flying monkeys are" \
  --use-f32

loaded the model in 5.364271244s
starting the inference loop
> 125.086509ms
1 token: 241 ' a'
> 50.189809ms
2 token: 1842 ' type'
> 43.385516ms
3 token: 275 ' of'
> 43.172109ms
4 token: 17564 ' creature'
> 43.394769ms
5 token: 325 ' that'
> 43.636299ms
6 token: 5741 ' appears'
> 43.482558ms
7 token: 272 ' in'
> 43.497175ms
8 token: 248 ' the'
> 43.305876ms
9 token: 204 ' '
> 43.491987ms
10 token: 6128 '193'
> 43.513766ms
11 token: 36 '9'
> 43.392516ms
12 token: 2675 ' film'
> 43.652215ms
13 token: 390 ' The'
> 43.437993ms
14 token: 31018 ' Wizard'
> 43.289842ms
15 token: 275 ' of'
> 43.413814ms
16 token: 21527 ' Oz'
> 43.901947ms
17 token: 25 '.'
> 43.608341ms
18 token: 1176 ' They'
> 43.747471ms
19 token: 362 ' are'
> 43.720673ms
20 token: 241 ' a'
> 43.7139ms
21 token: 1842 ' type'
> 43.626066ms
22 token: 275 ' of'
> 43.297449ms
23 token: 10184 ' flying'
> 43.483832ms
24 token: 17564 ' creature'
> 43.2593ms
25 token: 325 ' that'
> 43.471809ms
26 token: 304 ' is'
> 43.743894ms
27 token: 1042 ' used'
> 43.713613ms
28 token: 431 ' by'
> 43.570134ms
29 token: 248 ' the'
> 43.161667ms
30 token: 62099 ' Wicked'
> 43.216241ms
31 token: 32524 ' Witch'
> 43.711736ms
32 token: 275 ' of'
> 43.683056ms
33 token: 248 ' the'
> 43.305365ms
34 token: 3431 ' West'
> 43.647002ms
35 token: 271 ' to'
> 43.461744ms
36 token: 3033 ' attack'
> 43.197451ms
37 token: 38706 ' Dorothy'
> 43.236643ms
38 token: 273 ' and'
> 43.305153ms
39 token: 573 ' her'
> 43.42883ms
40 token: 2153 ' friends'
> 43.705694ms
41 token: 25 '.'
> 43.23486ms
42 token: 193 '
'
> 43.336939ms
43 token: 49 'F'
> 43.732691ms
44 token: 3985 'lying'
> 43.703105ms
45 token: 38889 ' monkeys'
> 43.555808ms
46 token: 362 ' are'
> 43.599027ms
47 token: 241 ' a'
> 43.321076ms
48 token: 1842 ' type'
> 43.394018ms
49 token: 275 ' of'
> 43.317828ms
50 token: 17564 ' creature'
> 43.388708ms
51 token: 325 ' that'
> 43.523916ms
52 token: 5741 ' appears'
> 43.784419ms
53 token: 272 ' in'
> 43.710184ms
54 token: 248 ' the'
> 43.647716ms
55 token: 204 ' '
> 43.570952ms
56 token: 6128 '193'
> 43.839569ms
57 token: 36 '9'
> 43.631927ms
58 token: 2675 ' film'
> 43.748854ms
59 token: 390 ' The'
> 43.351691ms
60 token: 31018 ' Wizard'
> 43.57853ms
61 token: 275 ' of'
> 43.332964ms
62 token: 21527 ' Oz'
> 43.326464ms
63 token: 25 '.'
> 43.72864ms
64 token: 1176 ' They'
> 43.90685ms
65 token: 362 ' are'
> 43.775395ms
66 token: 241 ' a'
> 43.894777ms
67 token: 1842 ' type'
> 43.225647ms
68 token: 275 ' of'
> 43.351282ms
69 token: 10184 ' flying'
> 43.768456ms
70 token: 17564 ' creature'
> 44.083289ms
71 token: 325 ' that'
> 43.403415ms
72 token: 304 ' is'
> 43.267355ms
73 token: 1042 ' used'
> 43.445115ms
74 token: 431 ' by'
> 43.805927ms
75 token: 248 ' the'
> 43.390881ms
76 token: 62099 ' Wicked'
> 43.35044ms
77 token: 32524 ' Witch'
> 43.337667ms
78 token: 275 ' of'
> 43.893648ms
79 token: 248 ' the'
> 43.708071ms
80 token: 3431 ' West'
> 43.682662ms
81 token: 271 ' to'
> 43.662521ms
82 token: 3033 ' attack'
> 43.533612ms
83 token: 38706 ' Dorothy'
> 43.537992ms
84 token: 273 ' and'
> 43.493205ms
85 token: 573 ' her'
> 43.559569ms
86 token: 2153 ' friends'
> 43.737254ms
87 token: 25 '.'
> 43.757527ms
88 token: 193 '
'
> 43.757487ms
89 token: 487 'The'
> 43.839315ms
90 token: 10184 ' flying'
> 43.894259ms
91 token: 38889 ' monkeys'
> 43.783155ms
92 token: 362 ' are'
> 43.719948ms
93 token: 241 ' a'
> 43.399592ms
94 token: 1842 ' type'
> 43.486655ms
95 token: 275 ' of'
> 43.629045ms
96 token: 17564 ' creature'
> 43.246949ms
97 token: 325 ' that'
> 43.392296ms
98 token: 5741 ' appears'
> 43.836782ms
99 token: 272 ' in'
> 43.724522ms
100 token: 248 ' the'
100 tokens generated (22.45401663769298 token/s)
----
 a type of creature that appears in the 1939 film The Wizard of Oz. They are a type of flying creature that is used by the Wicked Witch of the West to attack Dorothy and her friends.
Flying monkeys are a type of creature that appears in the 1939 film The Wizard of Oz. They are a type of flying creature that is used by the Wicked Witch of the West to attack Dorothy and her friends.
The flying monkeys are a type of creature that appears in the
----
```