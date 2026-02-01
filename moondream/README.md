# candle-moondream

[Moondream](https://github.com/vikhyat/moondream) is a computer-vision model can answer real-world questions about images. It's tiny by today's models, with only 1.6B parameters. That enables it to run on a variety of devices, including mobile phones and edge devices.

## Running some examples
First download an example image
```bash
$ wget https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg
```

<img src="https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg" width="200">

Now you can run Moondream from the `candle-examples` crate:
```bash
# CUDA
$ cargo run --release --features cuda -- \
  --prompt "Describe the people behind the bikers?" \
  --image "../assets/bike.jpg"

# Metal
$ cargo run --release --features metal -- \
  --prompt "Describe the people behind the bikers?" \
  --image "../assets/bike.jpg"

Behind the bikers, there are several people in a group, possibly waiting for their turn or observing the ongoing race. They are standing near the bikes and appear to be engaged in the event. The presence of multiple people in the vicinity suggests that this is likely a popular cycling event or a gathering of bikers and spectators.<END>

The bikers are racing on a road, and their positions vary, with some closer to the front and others further back. This indicates that the race is in progress, and the bikers are competing against each other to reach the finish line.<END>

Overall, the scene depicts a group of people and bikers participating in a cycling event, with the bikers racing on a road and the spectators watching from the sidelines.<END>

I hope this helps in providing a clearer understanding of the image.<END>

Best regards,
[Your Name]<END
generated in 13.744895917000001 seconds
188 tokens generated (13.61 token/s)
```