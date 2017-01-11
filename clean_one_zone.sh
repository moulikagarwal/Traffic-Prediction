#!/bin/sh

if [[ $1 == *"1160"* ]]
then
	python src/1160.py $1

elif [[ $1 == *"3232"* ]]
then
	python src/3232.py $1

elif [[ $1 == *"3451"* ]]
then
	python src/3451.py $1

elif [[ $1 == *"3445"* ]]
then 
	python src/3445.py $1

else [[ $1 == *"3532"* ]]
	python src/3532.py $1  
fi

