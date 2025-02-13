#!/bin/sh

export GOCACHE=/data/go-build
export PATH=$PATH:/data/data/com.termux/files/usr/bin
export GOMODCACHE=/data/go/pkg/mod

cd /data/data/com.termux/files/home/wmn_research/quic_phone_exp
chmod +x ./socket/quic_socket_phone.go

go run ./socket/quic_socket_phone.go -H $1 -d $2 -p $3 -t $4 -b $5 -l $6
# ./quic_phone_exp/socket/quic_socket_phone