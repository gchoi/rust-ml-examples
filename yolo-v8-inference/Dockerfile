# -- Build
FROM rust:latest AS build

WORKDIR /usr/src/
COPY . .

RUN cargo run --release -- ./assets/bike.jpg
RUN cp -r ~/.cache ./.cache

# -- Runtime
FROM ubuntu:latest

COPY --from=build /usr/src/assets/ /usr/local/bin/
COPY --from=build /usr/src/target/release/ /usr/local/bin/
COPY --from=build /usr/src/.cache /usr/local/bin/.cache
WORKDIR /usr/local/bin
RUN cp -r ./.cache ~/.cache
CMD ["candle-yolo-v8", "./bike.jpg"]