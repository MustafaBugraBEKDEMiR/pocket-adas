# 2023 prototype (archived)

This is the original graduation-project prototype (Hacettepe University, Geomatics
Engineering, 2023), kept for history. IP addresses have been replaced with placeholders.

```
Sensor Logger app --HTTP--> AWS API Gateway --> Lambda (Kafka producer)
                                                   |
                                         Kafka on EC2 (topic: sensor_data_consumer)
                                                   |
                                  consumer.py on PC / Pydroid: beep if |x|+|y|+|z| >= 15
```

Why it was replaced:

* **Gravity was never removed.** The rule `|x|+|y|+|z| >= 15` works on raw accelerometer
  values, which always contain ~9.81 m/s² of gravity. Depending on how the phone is held, a
  car standing still already reads 9.8–17, so alerts depended on the mount, not on the driving.
  The simulator shows this live ("2023 rule" line on the panel).
* **No direction.** A braking jolt, a sharp turn and a pothole all look the same to a single sum.
* **Cloud round-trip.** Kafka + EC2 + Lambda added cost and latency and needed mobile data.
  The new core runs on the device.
* The Keras model (≈65–75 % accuracy on the Kaggle *Driving Behavior* dataset) was dropped from
  the live loop in 2023 because its predictions were unstable; it was not reproducible.
