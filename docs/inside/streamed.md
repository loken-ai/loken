# A model larger than the server

```scenes
figure = "../anatomy/streamed.svg"
program = "../anatomy/streamed.scenes"
style = ["../anatomy/figures.css", "../anatomy/streamed.css"]
eyebrow = "lokend anatomy"
title = "A model larger than the server"
caption = "A model that does not fit"
hint = "Each square is an expert. The disk holds the model file, mapped. The RAM is the memory of the host, and its page cache holds the pages of the file that were read. The VRAM is the memory of the card. Large dots are bytes of experts on the move, small dark dots are rows of activations. The counts and sizes are an example drawn to show the shape."
soundtheme = "soft"
legendLabel = "Key:"
legend = [["st-disk", "an expert on the disk"], ["st-cache", "in the page cache"], ["st-card", "on the card"], ["st-pk", "bytes on the move"], ["st-row", "a row of activations"]]

[camera]
below = 640
over = 600
ratio = 1.4

[inlay]
pad = [0, 0]
below = 640
title = [[8, 6, 540, 84]]
now = [[690, 6, 992, 186]]
note = [[690, 6, 992, 186]]
step = [[8, 484, 992, 642]]

[[scene]]
key = "load"
keep = true
eyebrow = "Scene 1 of 6"
name = "Load"
title = "Loading a model that does not fit"
about = "A mixture of experts can be larger than the RAM of the host and the VRAM of the card together. Loken maps its file, keeps on the card what every token reads, and fills the rest of the card with the experts used most."
steps = [
  "The file is larger than the RAM and the VRAM together, so it is not read whole. The host maps it, and a page is read from the disk only when something touches it.",
  "The weights every token reads cross to the card once and stay there. They are read from the disk through the RAM, and copied into the VRAM before any routed expert.",
  "The room left in the VRAM is cut into expert slots. Before the first request, the slots are filled from the routing prior, the most used experts of every layer first. Each expert is read from the disk, copied to the card, and its pages leave the RAM.",
  "Every other expert stays on the disk. The host reads it in place when a token routes to it, and its page cache keeps the pages that were read.",
  "How many experts the page cache keeps for each layer moves while the server runs. It grows while the host has the room and the kernel reports no more memory stalls, and it falls back when it does.",
]
done = "Loaded, most of the file still on the disk"
now = ["Mapping the file", "Setting aside what every token reads", "Filling the slots from the prior", "The rest stays on the disk", "Sizing the page cache"]

[[scene]]
key = "time"
keep = true
eyebrow = "Scene 2 of 6"
name = "Over time"
title = "Experts moving over time"
about = "The experts of the file move between the disk, the RAM and the VRAM as tokens go by. Each square of the file takes the colour of where its expert is, and a counter gives where the experts of each token came from."
steps = [
  "The first tokens after the load. An expert on the card runs there. The others run on the host, and each is read from the disk the first time.",
  "The next tokens route to some of the same experts. Those are in the RAM already, so they are not read from the disk again, and the card counts each sighting.",
  "An expert seen often enough repays an upload. It moves from the RAM to the card, its pages leave the RAM, and the card trades out its least frequently routed expert.",
  "The conversation moves on to other experts. They are read from the disk, and the page cache turns over: the experts used last stay and the others give way.",
  "The new experts become frequent. They move to the card one by one, each trading out a colder expert, so the VRAM follows the experts the text keeps returning to.",
  "After many tokens, most of the experts a token routes to are on the card, some are in the RAM, and the disk is read only for the rare expert.",
] 
done = "The experts sit where the text needs them"
now = ["The first tokens", "Found in the RAM", "From the RAM to the card", "A new topic", "The card follows", "Settled"]

[[scene]]
key = "layer"
keep = true
eyebrow = "Scene 3 of 6"
name = "A layer"
title = "A layer of a token"
about = "A token crosses the layers one after the other. In each layer, the attention runs on the card, the router names the experts, and each expert runs on the card or on the host."
steps = [
  "A token runs the attention of a layer on the card, from the weights kept there.",
  "A row of the stream crosses to the host, where the router names the experts.",
  "An expert kept in the VRAM runs on the card. Its row crosses the bus, the card computes, and the product comes back.",
  "An expert that is not on the card runs on the host cores, at the same time. It reads its weights from the mapping in place: they are never dequantised or copied.",
  "The products of both sides are summed with the router's weights, and the stream goes on in the card.",
]
done = "The stream goes on to the next layer"
now = ["Attention on the card", "The router names the experts", "Experts kept on the card", "Experts read in place by the host", "Summing the products"]

[[scene]]
key = "disk"
keep = true
eyebrow = "Scene 4 of 6"
name = "Disk reads"
title = "Reading from the disk"
about = "What the host needs and the RAM does not hold is read from the disk. The reads are made one at a time on a fault, or ahead of time by reader threads."
steps = [
  "A page that is not in the RAM is read from the disk when a core touches it. A fault reads 1 expert at a time.",
  "Reader threads read the experts of the next layers before they are needed. They guess them from the stream the token has by then.",
  "How many layers ahead they read is moved by the decode's own timing. A controller tries each neighbour for a window and keeps one only when the tokens run faster.",
  "The page cache keeps the experts used last. The pages of the others are dropped only while the kernel reports memory stalls, and stay otherwise.",
]
done = "The pages are where the next token needs them"
deeper = [{ text = "Streamed mixtures, how each quantity is derived", href = "../CONFIGURATION.md#streamed-mixtures-deepseek-v41" }]
now = ["A page fault", "Reading ahead", "Tuning the depth", "Keeping or dropping pages"]

[[scene]]
key = "card"
keep = true
eyebrow = "Scene 5 of 6"
name = "To the card"
title = "An expert moves to the card"
about = "An expert that the host runs again and again is worth the card's room. The card counts, copies it on a stream of its own, and trades out its coldest expert when it is full."
steps = [
  "An expert that is not on the card runs on the host, and the card counts each time it sees it.",
  "Once an expert was seen as many times as an upload needs uses to pay for itself, it is queued for the copy thread. An upload costs a measured time, and each use on the card saves the measured time of the host.",
  "The copy thread sends the expert on a stream of its own. The copy goes beside the card's compute and off the token's path, so the token runs the expert on the host.",
  "The card now holds the copy, so the host pages that were read for it are released. They would take the room of experts that the host still needs.",
  "When the VRAM is full, a newcomer takes the place of the least frequently routed expert. It never takes the place of an expert used within the layers of this token, and only of one colder than itself.",
  "The heat of an evicted expert outlives its slot. It comes back at the count it earned, so the slots settle on the hottest experts instead of churning.",
]
done = "The slots follow the hottest experts"
now = ["Counting the sightings", "The upload pays back", "Copying on its own stream", "Releasing the host pages", "Trading out the coldest", "The heat outlives the slot"]

[[scene]]
key = "prompt"
keep = true
eyebrow = "Scene 6 of 6"
name = "A prompt"
title = "A prompt through the experts"
about = "A prompt routes to many experts in each layer. Each expert goes where it costs least, and the copies for a layer never take a kept expert's place."
steps = [
  "A prompt routes its tokens to many experts in each layer. The layer starts when the router has named them.",
  "Each expert goes where it costs least. An expert kept on a card runs there from the card's copy. Another runs on a card when a copy of it costs less than its rows cost the host, and stays on the host otherwise.",
  "The copies are read from the file into a page-locked slab, in runs of adjacent experts, around the page cache.",
  "The card computes the experts it keeps and the copies, and the host computes its own. All of it happens at the same time.",
  "A copy made for the layer is let go after it unless the card has the room free to keep it. Nothing is evicted for it: it never takes a kept expert's place.",
  "When the cards take no part of the prompt and it routes to most of the experts, whole layers are read in. The next layer is read into the RAM while the current layer computes.",
]
done = "Prompt read, layer by layer"
now = ["Naming the experts", "Choosing card or host", "Reading adjacent experts", "Everything at the same time", "Letting the copies go", "Reading the next layer ahead"]
```

A mixture whose experts stay in the file's mapping is served this way.
The experts a card keeps, the room for copies and the number of experts the page cache keeps for each layer all come from measurement at load.
See [Expert cache of a mixture held in the file's mapping](../CONFIGURATION.md#expert-cache-of-a-mixture-held-in-the-files-mapping) for how each quantity is derived.

DeepSeek V4.1 and the other wide-residual models take the same generic route, with their own tiers:

- The host maps the file, and the cards keep what every token reads.
- The card tier fills the rest of the cards with the experts the routing heat ranks first, at load. It takes no expert on while tokens decode.
- The host tier keeps in the page cache the experts the decoded rows route to most or last, and drops the pages of every other expert once computed.

See [Streamed mixtures](../CONFIGURATION.md#streamed-mixtures-deepseek-v41) for what the `fidelity` setting trades.
