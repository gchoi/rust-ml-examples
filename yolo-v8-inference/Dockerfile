# ------------------------------------------------
# - Build
# ------------------------------------------------
FROM rust:latest AS build

WORKDIR /usr/src/
COPY . .

RUN cargo run --release -- ./assets/bike.jpg
RUN cp -r ~/.cache ./.cache

# ------------------------------------------------
# - Runtime
# ------------------------------------------------
FROM ubuntu:latest

COPY --from=build /usr/src/target/release/ /usr/local/bin/
COPY --from=build /usr/src/.cache /usr/local/bin/.cache

WORKDIR /usr/local/bin
RUN mkdir -p images
RUN cp -r ./.cache ~/.cache
RUN rm -rf .cache
RUN rm -rf build
RUN rm -r .cargo-lock
RUN rm -r .fingerprint
RUN rm -rf deps
RUN rm -rf examples
RUN rm -rf incremental
ENTRYPOINT ["candle-yolo-v8"]