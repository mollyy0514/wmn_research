// Open socket for every client phone
// Since we might implement both UL&DL in the future (we only use UL for now),
// I still assign 2 ports for each device, ports[0] for UL, ports[1] for DL

package main

import (
	"context"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mollyy0514/quic-go"
	"github.com/mollyy0514/quic-go/logging"
	"github.com/mollyy0514/quic-go/qlog"
)

// const SERVER = "127.0.0.1"
// const SERVER = "192.168.1.78"
const SERVER = "140.112.20.183"

var BITRATE int
var PACKET_LEN int
var SLEEPTIME float64
var PORT_UL int
var PORT_DL int

func main() {
	/* ---------- USER SETTING --------- */
	// Define command-line flags
	_host := flag.String("H", "140.112.20.183", "server ip address")
	_devices := flag.String("d", "sm00", "list of devices (space-separated)")
	_ports := flag.String("p", "5200,5201", "ports to bind (space-separated)")
	_bitrate := flag.String("b", "0", "target bitrate in bits/sec (0 for unlimited)")
	_length := flag.Int("l", 1223, "length of buffer to read or write in bytes (packet size)")
	_duration := flag.Int("t", 300, "time in seconds to transmit for (default 1 hour = 3600 secs)")
	// Parse command-line arguments
	flag.Parse()
	fmt.Printf("INFO: %s %s %s %s %d %d \n", *_host, *_devices, *_ports, *_bitrate, *_length, *_duration)

	duration := *_duration
	portsList := strings.Split(*_ports, ",")

	if len(portsList) == 2 {
		PORT_UL, _ = strconv.Atoi(portsList[0])
		PORT_DL, _ = strconv.Atoi(portsList[1])
	} else {
		fmt.Println("port missing!")
	}
	var serverAddr_ul string = fmt.Sprintf("%s:%d", *_host, PORT_UL)
	var serverAddr_dl string = fmt.Sprintf("%s:%d", *_host, PORT_DL)

	PACKET_LEN = *_length
	bitrate_string := *_bitrate

	num, unit := bitrate_string[:len(bitrate_string)-1], bitrate_string[len(bitrate_string)-1:]
	if unit == "k" {
		numVal, _ := strconv.ParseFloat(num, 64)
		BITRATE = int(numVal * 1e3)
	} else if unit == "M" {
		numVal, _ := strconv.ParseFloat(num, 64)
		BITRATE = int(numVal * 1e6)
	} else {
		numVal, _ := strconv.ParseFloat(num, 64)
		BITRATE = int(numVal)
	}
	if BITRATE != 0 {
		expected_packet_per_sec := float64(BITRATE) / (float64(PACKET_LEN) * 8)
		SLEEPTIME = 1000 / expected_packet_per_sec
	} else {
		SLEEPTIME = 0
	}
	/* ---------- USER SETTING --------- */

	// wait for tcpdump
	// sleep 1 sec to ensure the whle handshake process is captured
	time.Sleep(1 * time.Second) 

	// get current time
	currentTime := time.Now()
	y := currentTime.Year()
	m := currentTime.Month()
	d := currentTime.Day()
	h := currentTime.Hour()
	n := currentTime.Minute()
	date := fmt.Sprintf("%02d%02d%02d", y, m, d)

	// create directory in the name of current date
	folderDate := fmt.Sprintf("%02d-%02d-%02d", y, m, d)
	basePath := "/sdcard/experiment_log"
	logFileDirPath := filepath.Join(basePath, folderDate)
	if _, err := os.Stat(logFileDirPath); os.IsNotExist(err) {
		err = os.MkdirAll(logFileDirPath, 0755) // 0755 is a common permission setting
		if err != nil {
			fmt.Println("Error creating directory:", err)
			return
		}
		fmt.Println("Directory created:", logFileDirPath)
	} else {
		fmt.Println("Directory already exists:", logFileDirPath)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	for i := 0; i < 2; i++ {
		go func(i int) { // capture packets in client side
			if i == 0 { // UPLINK
				// set generate configs
				keyLogFileUl := fmt.Sprintf("%s/tls_key_%s_%02d%02d_%02d.log", logFileDirPath, date, h, n, PORT_UL)
				var keyLogUl io.Writer
				if len(keyLogFileUl) > 0 {
					f, err := os.Create(keyLogFileUl)
					if err != nil {
						log.Fatal(err)
					}
					defer f.Close()
					keyLogUl = f
				}
				poolUl, err := x509.SystemCertPool()
				if err != nil {
					log.Fatal(err)
				}
				tlsConfig := GenTlsConfig()
				tlsConfig.RootCAs = poolUl
				tlsConfig.KeyLogWriter = keyLogUl

				quicConfig := GenQuicConfig(PORT_UL, logFileDirPath)

				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second) // 3s handshake timeout
				defer cancel()
				// connect to server IP. Session is like the socket of TCP/IP
				session_ul, err := quic.DialAddr(ctx, serverAddr_ul, tlsConfig, &quicConfig)
				if err != nil {
					fmt.Println("err: ", err)
				}
				defer session_ul.CloseWithError(quic.ApplicationErrorCode(501), "hi you have an error")
				// create a stream_ul
				// context.Background() is similar to a channel, giving QUIC a way to communicate
				stream_ul, err := session_ul.OpenStreamSync(context.Background())
				if err != nil {
					log.Fatal(err)
				}
				defer stream_ul.Close()

				Client_send(stream_ul, duration)
				time.Sleep(1 * time.Second)
				session_ul.CloseWithError(0, "ul times up")
				/* ---------- TCPDUMP ---------- */
				// Close_client_tcpdump(subp1)
				/* ---------- TCPDUMP ---------- */
			} else { // DOWNLINK
				// set generate configs
				keyLogFileDl := fmt.Sprintf("%s/tls_key_%s_%02d%02d_%02d.log", logFileDirPath, date, h, n, PORT_DL)
				var keyLogDl io.Writer
				if len(keyLogFileDl) > 0 {
					f, err := os.Create(keyLogFileDl)
					if err != nil {
						log.Fatal(err)
					}
					defer f.Close()
					keyLogDl = f
				}
				poolDl, err := x509.SystemCertPool()
				if err != nil {
					log.Fatal(err)
				}
				tlsConfig := GenTlsConfig()
				tlsConfig.RootCAs = poolDl
				tlsConfig.KeyLogWriter = keyLogDl

				quicConfig := GenQuicConfig(PORT_DL, logFileDirPath)

				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second) // 3s handshake timeout
				defer cancel()
				// connect to server IP. Session is like the socket of TCP/IP
				session_dl, err := quic.DialAddr(ctx, serverAddr_dl, tlsConfig, &quicConfig)
				if err != nil {
					fmt.Println("err: ", err)
				}
				defer session_dl.CloseWithError(quic.ApplicationErrorCode(501), "hi you have an error")
				// create a stream_dl
				// context.Background() is similar to a channel, giving QUIC a way to communicate
				stream_dl, err := session_dl.OpenStreamSync(context.Background())
				if err != nil {
					log.Fatal(err)
				}
				defer stream_dl.Close()

				// Open or create a file to store the floats in TXT format
				filepath := fmt.Sprintf("%s/time_%s_%02d%02d_%d.txt", logFileDirPath, date, h, n, PORT_DL)
				timeFile, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if err != nil {
					fmt.Println("Error opening file:", err)
					return
				}
				defer timeFile.Close()

				var message []byte
				t := time.Now().UnixNano() // Time in milliseconds
				fmt.Println("client create time: ", t)
				datetimedec := uint32(t / 1e9) // Extract seconds from milliseconds
				microsec := uint32(t % 1e9)    // Extract remaining microseconds
				message = append(message, make([]byte, 4)...)
				binary.BigEndian.PutUint32(message[:4], datetimedec)
				message = append(message, make([]byte, 4)...)
				binary.BigEndian.PutUint32(message[4:8], microsec)
				SendStartPacket(stream_dl, message)

				prev_receive := 0
				time_slot := 1
				seq := 1
				for {
					buf := make([]byte, PACKET_LEN)
					ts, err := Client_receive(stream_dl, buf)
					seq++
					if ts == -115 {
						session_dl.CloseWithError(0, "dl times up")
					}
					if err != nil {
						return
					}
					// fmt.Printf("client received: %f\n", ts)
					if time.Since(currentTime) > time.Second*time.Duration(time_slot) {
						fmt.Printf("%d [%d-%d] receive %d\n", PORT_DL, time_slot-1, time_slot, seq-prev_receive)
						time_slot += 1
						prev_receive = seq
					}

					// Write the timestamp as a string to the text file
					_, err = timeFile.WriteString(fmt.Sprintf("%f\n", ts))
					if err != nil {
						fmt.Println("Error writing to file:", err)
						return
					}
				}

			}
		}(i)
	}
	wg.Wait()
}

func GenTlsConfig() *tls.Config {
	// set TLS
	return &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{"h3"},
	}
}

func GenQuicConfig(port int, logFileDirPath string) quic.Config {
	return quic.Config{
		Allow0RTT: true,
		Tracer: func(ctx context.Context, p logging.Perspective, connID quic.ConnectionID) *logging.ConnectionTracer {
			role := "server"
			if p == logging.PerspectiveClient {
				role = "client"
			}
			currentTime := time.Now()
			y := currentTime.Year()
			m := currentTime.Month()
			d := currentTime.Day()
			h := currentTime.Hour()
			n := currentTime.Minute()
			date := fmt.Sprintf("%02d%02d%02d", y, m, d)
			filename := fmt.Sprintf("%s/log_%s_%02d%02d_%d_%s.qlog", logFileDirPath, date, h, n, port, role)
			f, err := os.Create(filename)
			if err != nil {
				fmt.Println(err)
				fmt.Println("Cannot generate qlog file.")
			}
			// handle the error
			return qlog.NewConnectionTracer(f, p, connID)
		},
	}
}

func Client_create_packet(euler uint32, pi uint32, datetimedec uint32, microsec uint32, seq uint32) []byte {
	var message []byte
	message = append(message, make([]byte, 4)...)
	binary.BigEndian.PutUint32(message[:4], euler)
	message = append(message, make([]byte, 4)...)
	binary.BigEndian.PutUint32(message[4:8], pi)
	message = append(message, make([]byte, 4)...)
	binary.BigEndian.PutUint32(message[8:12], datetimedec)
	message = append(message, make([]byte, 4)...)
	binary.BigEndian.PutUint32(message[12:16], microsec)
	message = append(message, make([]byte, 4)...)
	binary.BigEndian.PutUint32(message[16:20], seq)

	// add random additional data to 250 bytes
	msgLength := len(message)
	if msgLength < PACKET_LEN {
		randomBytes := make([]byte, PACKET_LEN-msgLength)
		rand.Read(randomBytes)
		message = append(message, randomBytes...)
	}

	return message
}

func SendStartPacket(stream quic.Stream, message []byte) {
	_, err := stream.Write(message)
	if err != nil {
		log.Fatal(err)
	}
}

func Client_send(stream quic.Stream, duration int) {
	prev_transmit := 0
	time_slot := 1
	seq := 1
	start_time := time.Now()
	euler := 271828
	pi := 31415926
	next_transmission_time := float64(start_time.UnixNano()) / 1e6
	for time.Since(start_time) <= time.Second*time.Duration(duration) {
		for float64(time.Now().UnixNano())/1e6 < next_transmission_time {
			// t = time.Now().UnixNano()
		}
		next_transmission_time += SLEEPTIME
		t := time.Now().UnixNano() // Time in milliseconds
		// fmt.Println("client sent: ", t)	// print out the time that sent to server in every packet
		datetimedec := uint32(t / 1e9) // Extract seconds from milliseconds
		microsec := uint32(t % 1e9)    // Extract remaining microseconds

		// var message []byte
		message := Client_create_packet(uint32(euler), uint32(pi), datetimedec, microsec, uint32(seq))
		SendStartPacket(stream, message)

		if time.Since(start_time) > time.Second*time.Duration(time_slot) {
			fmt.Printf("%d [%d-%d] transmit %d\n", PORT_UL, time_slot-1, time_slot, seq-prev_transmit)
			time_slot += 1
			prev_transmit = seq
		}
		seq++
	}
}

func Client_receive(stream quic.Stream, buf []byte) (float64, error) {
	_, err := stream.Read(buf)
	tsSeconds := binary.BigEndian.Uint32(buf[8:12])
	tsMicroseconds := binary.BigEndian.Uint32(buf[12:16])
	var ts float64
	if tsSeconds == 115 && tsMicroseconds == 115 {
		return -115, err
	} else {
		ts = float64(tsSeconds) + float64(tsMicroseconds)/1e9
	}

	if err != nil {
		return -1103, err
	}

	return ts, err
}
