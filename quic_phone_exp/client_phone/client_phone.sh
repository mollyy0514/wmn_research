#!/bin/sh

export GOCACHE=/data/go-build
export PATH=$PATH:/data/data/com.termux/files/usr/bin
export GOMODCACHE=/data/go/pkg/mod

cd /data/data/com.termux/files/home/wmn_research
chmod +x ./quic_phone_exp/socket/quic_socket_phone.go

go run ./quic_phone_exp/socket/quic_socket_phone.go -d $1 -p $2 -t $3 -b $4 -l $5
# ./quic_phone_exp/socket/quic_socket_phone