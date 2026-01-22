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

COPY ./assets/ /usr/local/bin/assets/
COPY --from=build /usr/src/target/release/ /usr/local/bin/
COPY --from=build /usr/src/.cache /usr/local/bin/

WORKDIR /usr/local/bin
RUN cp -r ./.cache ~/.cache
RUN
CMD ["candle-yolo-v8"]