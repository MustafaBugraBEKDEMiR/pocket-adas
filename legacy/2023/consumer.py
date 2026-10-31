from kafka import KafkaConsumer
from kafka import TopicPartition , OffsetAndMetadata
import json
from playsound import playsound

ses_dosyasi = "beep.mp3"

consumer = KafkaConsumer ('sensor_data_consumer',bootstrap_servers = ['KAFKA_BROKER_IP:9092'],
value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                          group_id='acceleration_test',
                          auto_offset_reset='earliest',
                          enable_auto_commit =False)


for message in consumer:
    data=message.value
    x_accl=data['acc_x']
    y_accl = data['acc_y']
    z_accl = data['acc_z']
    mag=(abs(x_accl)+abs(y_accl)+abs(z_accl))
    if(mag>=15):
        print("YOU ARE IN DANGER")
        playsound(ses_dosyasi)
    else:
        print("YOU ARE IN SAFE")
    tp=TopicPartition(message.topic,message.partition)
    om = OffsetAndMetadata(message.offset+1, message.timestamp)
    consumer.commit({tp:om})
