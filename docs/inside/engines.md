# Loken beside other engines

```scenes
figure = "../anatomy/engines.svg"
program = "../anatomy/engines.scenes"
style = ["../anatomy/figures.css", "../anatomy/engines.css"]
eyebrow = "lokend anatomy"
title = "The same server, 3 engines"
caption = "vLLM, Ollama and loken"
hint = "A layer is a slab of the strip, and a request a gold slab. A square is a place on a device. The rows of vLLM and Ollama show what their code and documentation say, with the settings they name. The numbers of layers and places are an example."
soundtheme = "soft"
legendLabel = "Key:"
legend = [["eg-card", "on a card"], ["eg-host", "in host memory"], ["eg-remote", "on another machine"], ["eg-stop", "stopped"]]

[camera]
below = 640
over = 600
ratio = 1.4

[inlay]
pad = [0, 0]
below = 640
title = [[8, 6, 520, 70]]
now = [[540, 6, 792, 92]]
note = [[540, 6, 792, 92]]
step = [[8, 556, 792, 634]]

[[scene]]
key = "room"
keep = true
eyebrow = "Scene 1 of 4"
name = "Not enough room"
title = "A model larger than the card"
about = "The same model on the same server, a card and host memory, with a model larger than the card. Each engine decides differently what happens."
steps = [
  "The model has 8 layers and the card holds 5. The host has room for the others.",
  "vLLM loads the weights on the card. Host memory holds weights only when `cpu_offload_gb` is set, and its default is 0.",
  "With `cpu_offload_gb` set by the owner, a part of the weights is read from host memory at each forward pass. That asks for a fast link between the host and the card.",
  "At start, vLLM also checks that the KV cache of 1 request at the model's maximum length fits in what is left. If it does not, the start stops with an error that names the settings to change.",
  "Ollama runs the model in llama-server, which decides the number of layers on the card unless `num_gpu` is set. The other layers run on the host.",
  "Loken fills the card with the layers it holds, and the host takes only the layers that no card holds.",
  "Only if that placement fails with an out-of-memory error does loken try the next rung of a ladder. The rungs go from an even split, to the cards at a smaller share with the host, to the host alone.",
]
done = "Each engine has answered"
now = ["The model and the server", "vLLM, on the card", "vLLM, with the host", "vLLM, the KV check", "Ollama, decided at load", "Loken, card first, host for the rest", "Loken, after a failure"]

[[scene]]
key = "cards"
keep = true
eyebrow = "Scene 2 of 4"
name = "Several cards"
title = "A model across 2 cards"
about = "The model fits no card alone and fits the 2 cards together. Card A is faster than card B."
steps = [
  "No card holds the 8 layers alone, and the 2 cards together do. Card A is faster than card B.",
  "In vLLM, `tensor_parallel_size` and `pipeline_parallel_size` are set by the owner, and both default to 1.",
  "With `tensor_parallel_size` set to 2, each layer is split across both cards.",
  "Ollama loads the model on 1 GPU when it fits there entirely, and spreads it across all the available GPUs when it does not.",
  "Loken ranks the cards by compute rate and fills them in that order until every layer has a card. The layers are then shared out by rate, so the faster card holds more.",
  "When the model fits no card and every layer can be split, loken splits every layer across the cards if that is predicted faster.",
]
done = "Each engine has answered"
now = ["2 cards, a model that fits neither", "vLLM, sizes of 1", "vLLM, tensor parallel", "Ollama, spread", "Loken, ranked by rate", "Loken, split when faster"]

[[scene]]
key = "machines"
keep = true
eyebrow = "Scene 3 of 4"
name = "Several machines"
title = "A model across 2 machines"
about = "The model is larger than the card of a machine, and a second machine is on the network."
steps = [
  "A model that no card holds alone, on 2 machines linked by a network.",
  "In vLLM, the owner sets `nnodes` and the parallel sizes, and picks the `ray` or the `mp` executor backend.",
  "Ollama reads the video memory and the system memory of the machine it runs on, and places the model there.",
  "In loken, any machine of the cluster can receive a request. It predicts the completion time here and on its peer, and sends the whole request to the peer when the peer wins by a margin.",
  "A model larger than a machine can be cut across 2 machines. Each machine runs 1 run of layers, on its cards first, and the cut is the one with the shortest predicted time for 1 token.",
]
done = "Each engine has answered"
now = ["2 machines on a network", "vLLM, nodes by hand", "Ollama, its own machine", "Loken, forwarding the request", "Loken, cutting the model"]

[[scene]]
key = "parallel"
keep = true
eyebrow = "Scene 4 of 4"
name = "Requests together"
title = "3 requests at once"
about = "3 requests reach the same model together."
steps = [
  "3 requests reach the same model together.",
  "vLLM batches the running requests in its scheduler, step by step, up to `max_num_seqs` sequences, 128 by default.",
  "Ollama serves `OLLAMA_NUM_PARALLEL` requests per model at a time, 1 by default, and queues the others. The memory for the context grows with each parallel request.",
  "Loken batches the requests that overlap in its worker. A lone request takes the path, serial or worker, that serves a request like the latest lone one sooner. The serial path drafts. When another request arrives during a serial decode, the decode moves into the worker at its next step.",
]
done = "Each engine has answered"
now = ["3 requests together", "vLLM, continuous batching", "Ollama, parallel slots", "Loken, serial or batch"]
```

The rows of vLLM and Ollama are read from their code and their documentation. The settings named in the scenes are the ones that decide each case.
