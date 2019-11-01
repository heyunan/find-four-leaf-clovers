#include <Servo.h>

#define SW_PIN 2
#define JETSON_PIN 4
#define SERVO1 10
#define SPK_PIN 12

Servo servo1;
int sw_val;
int jetson_val;

void setup() {
  // put your setup code here, to run once:
  pinMode(SPK_PIN, OUTPUT);
  pinMode(SW_PIN, INPUT);
  pinMode(JETSON_PIN, INPUT);
  servo1.attach(SERVO1);

}

void loop() {
  // put your main code here, to run repeatedly:

  sw_val = digitalRead(SW_PIN);
  jetson_val = digitalRead(JETSON_PIN);
  
  if(sw_val==HIGH || jetson_val==HIGH){
    tone(SPK_PIN, 500, 100);
    delay(300);
    servo1.write(120); 
    delay(1600);
    servo1.write(100);
    delay(1000);
  }else{
   // tone(SPK_PIN, 500, 100);
    servo1.write(100);
    //delay(1000);
  }
}
