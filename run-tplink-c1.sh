./cds detect \
	--uri 'rtsp://admin:cwvqYgGn4vjGN3oKYdVBj@c1.dpdtech.com:554/h264Preview_01_main' \
	--model-path artifacts/models/communitycats-prod-20260228-193539/exports/best.mlpackage \
	--imgsz 640 \
	--confidence 0.75 \
	--nms 0.6 \
	--labels-path  config/classes-communitycats-prod-20260228-193539.txt 
