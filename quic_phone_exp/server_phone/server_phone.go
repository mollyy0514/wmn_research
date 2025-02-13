package main

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"encoding/pem"
	"flag"
	"fmt"
	"log"
	"math/big"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"quic_exp/devices"

	"github.com/mollyy0514/quic-go"
	"github.com/mollyy0514/quic-go/logging"
	"github.com/mollyy0514/quic-go/qlog"
)

const SERVER = "0.0.0.0"

var BITRATE int
var PACKET_LEN int
var SLEEPTIME float64

func main() {

	fmt.Println("Starting server...")
	/* ---------- USER SETTING --------- */
	// Define command-line flags
	_password := flag.String("p", "", "password")
	_devices := flag.String("d", "sm00", "list of devices (space-separated)")
	_bitrate := flag.String("b", "0", "target bitrate in bits/sec (0 for unlimited)")
	_length := flag.Int("l", 1223, "length of buffer to read or write in bytes (packet size)")
	_duration := flag.Int("t", 300, "time in seconds to transmit for (default 1 hour = 3600 secs)")
	flag.Parse()
	if *_password == "" {
		fmt.Print("Please enter password for tcpdump.")
		os.Exit(1)
	}

	_devices_string := *_devices
	devicesList := Get_devices(_devices_string)
	portsList := Get_Port(devicesList)
	print("deviceCnt: ", len(portsList), "\n")
	duration := *_duration
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

	/* ---------- TCPDUMP ---------- */
	// for i := 0; i < len(portsList); i++ {
	// 	Start_server_tcpdump(*_password, portsList[i][0])
	// 	Start_server_tcpdump(*_password, portsList[i][1])
	// }
	/* ---------- TCPDUMP ---------- */
	currentTime := time.Now()
	y := currentTime.Year()
	m := currentTime.Month()
	d := currentTime.Day()

	// create directory in the name of current date
	folderDate := fmt.Sprintf("%02d-%02d-%02d", y, m, d)
	basePath := "/home/wmnlab/temp/QUIC_temp"
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

	// Sync between goroutines.
	var wg sync.WaitGroup
	for i := 0; i < len(portsList); i++ {
		wg.Add(2)
		defer wg.Done()

		go EchoQuicServer(SERVER, portsList[i][0], true, duration, logFileDirPath)
		go EchoQuicServer(SERVER, portsList[i][1], false, duration, logFileDirPath)
	}
	wg.Wait()
}

func HandleQuicStream_ul(stream quic.Stream, quicPort int, duration int, logFileDirPath string) {
	// Open or create a file to store the floats in JSON format
	currentTime := time.Now()
	y := currentTime.Year()
	m := currentTime.Month()
	d := currentTime.Day()
	h := currentTime.Hour()
	n := currentTime.Minute()
	date := fmt.Sprintf("%02d%02d%02d", y, m, d)
	
	timeDirPath := filepath.Join(logFileDirPath, "time_file")
	err := os.MkdirAll(timeDirPath, os.ModePerm)
	if err != nil {
		log.Fatal("Error creating directory:", err)
	}
	filepath := fmt.Sprintf("%s/time_%s_%02d%02d_%d.txt", timeDirPath, date, h, n, quicPort)
	timeFile, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return
	}
	defer timeFile.Close()

	prev_receive := 0
	time_slot := 1
	seq := 1
	for {
		buf := make([]byte, PACKET_LEN)
		ts, err := Server_receive(stream, buf)
		seq++
		if err != nil {
			return
		}
		// fmt.Printf("Received %d: %f\n", quicPort, ts)
		if time.Since(currentTime) > time.Second*time.Duration(time_slot) {
			fmt.Printf("%d [%d-%d] receive %d\n", quicPort, time_slot-1, time_slot, seq-prev_receive)
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

func HandleQuicStream_dl(stream quic.Stream, quicPort int, duration int) {
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
		t := time.Now().UnixNano()
		next_transmission_time += SLEEPTIME
		// fmt.Println("server sent:", t)	// print out the time that sent to client in every packet
		datetimedec := uint32(t / 1e9) // Extract seconds from milliseconds
		microsec := uint32(t % 1e9)    // Extract remaining microseconds

		// var message []byte
		message := Server_create_packet(uint32(euler), uint32(pi), datetimedec, microsec, uint32(seq))
		Server_transmit(stream, message)

		if time.Since(start_time) > time.Second*time.Duration(time_slot) {
			fmt.Printf("%d [%d-%d] transmit %d\n", quicPort, time_slot-1, time_slot, seq-prev_transmit)
			time_slot += 1
			prev_transmit = seq
		}
		seq++
	}
	message := Server_create_packet(uint32(euler), uint32(pi), 115, 115, uint32(seq))
	Server_transmit(stream, message)
}

func HandleQuicSession(sess quic.Connection, quicPort int, ul bool, duration int, logFileDirPath string) {
	for {
		// create a stream to receive message, and also create a channel for communication
		stream, err := sess.AcceptStream(context.Background())
		if err != nil {
			fmt.Println(err)
			return // Using panic here will terminate the program if a new connection has not come in in a while, such as transmitting large file.
		}

		if ul {
			go HandleQuicStream_ul(stream, quicPort, duration, logFileDirPath)
		} else {
			go HandleQuicStream_dl(stream, quicPort, duration)
		}
	}
}

// Start a server that echos all data on top of QUIC
func EchoQuicServer(host string, quicPort int, ul bool, duration int, logFileDirPath string) error {
	qlogDirPath := filepath.Join(logFileDirPath, "server_qlog")
	err := os.MkdirAll(qlogDirPath, os.ModePerm)
	if err != nil {
		log.Fatal("Error creating directory:", err)
	}
	quicConfig := quic.Config{
		KeepAlivePeriod: time.Minute * 5,
		EnableDatagrams: true,
		Allow0RTT:       true,
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

			filename := fmt.Sprintf("%s/log_%s_%02d%02d_%d_%s.qlog", qlogDirPath, date, h, n, quicPort, role)
			f, err := os.Create(filename)
			if err != nil {
				fmt.Println("cannot generate qlog file")
			}
			// handle the error
			return qlog.NewConnectionTracer(f, p, connID)
		},
	}
	listener, err := quic.ListenAddr(fmt.Sprintf("%s:%d", host, quicPort), generateTLSConfig(quicPort, logFileDirPath), &quicConfig)
	if err != nil {
		return err
	}

	fmt.Printf("Started QUIC server! %s:%d\n", host, quicPort)

	for {
		// create a session
		sess, err := listener.Accept(context.Background())
		fmt.Printf("Accepted Connection! %s\n", sess.RemoteAddr())

		if err != nil {
			return err
		}

		go HandleQuicSession(sess, quicPort, ul, duration, logFileDirPath)
	}
}

// Setup a bare-bones TLS config for the server
func generateTLSConfig(quicPort int, logFileDirPath string) *tls.Config {
	key, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		panic(err)
	}
	template := x509.Certificate{SerialNumber: big.NewInt(1)}
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
	if err != nil {
		panic(err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})

	currentTime := time.Now()
	y := currentTime.Year()
	m := currentTime.Month()
	d := currentTime.Day()
	h := currentTime.Hour()
	n := currentTime.Minute()
	date := fmt.Sprintf("%02d%02d%02d", y, m, d)

	keyDirPath := filepath.Join(logFileDirPath, "tls_key")
	err = os.MkdirAll(keyDirPath, os.ModePerm)
	if err != nil {
		log.Fatal("Error creating directory:", err)
	}
	keyFileName := fmt.Sprintf("%s/tls_key_%s_%02d%02d_%02d.log", keyDirPath, date, h, n, quicPort)
	kl, _ := os.OpenFile(keyFileName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)

	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		panic(err)
	}
	return &tls.Config{
		Certificates: []tls.Certificate{tlsCert},
		NextProtos:   []string{"h3"},
		KeyLogWriter: kl,
	}
}

func Start_server_tcpdump(password string, port int, logFileDirPath string) {
	currentTime := time.Now()
	y := currentTime.Year()
	m := currentTime.Month()
	d := currentTime.Day()
	h := currentTime.Hour()
	n := currentTime.Minute()
	date := fmt.Sprintf("%02d%02d%02d", y, m, d)
	
	pcapDirPath := filepath.Join(logFileDirPath, "server_pcap")
	err := os.MkdirAll(pcapDirPath, os.ModePerm)
	if err != nil {
		log.Fatal("Error creating directory:", err)
	}
	filepath := fmt.Sprintf("%s/capturequic_s_%s_%02d%02d_%d.pcap", pcapDirPath, date, h, n, port)
	command := fmt.Sprintf("echo %s | sudo -S tcpdump port %d -w %s", password, port, filepath)
	cmd := exec.Command("sh", "-c", command)
	err = cmd.Start()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("tcpdump start for: %02d \n", port)
}

func Get_devices(_devices_string string) []string {
	var devicesList []string
	// var serialsList []string
	if strings.Contains(_devices_string, "-") {
		pmodel := _devices_string[:2]
		start, _ := strconv.Atoi(_devices_string[2:4])
		stop, _ := strconv.Atoi(_devices_string[5:7])
		for i := start; i <= stop; i++ {
			_dev := fmt.Sprintf("%s%02d", pmodel, i)
			devicesList = append(devicesList, _dev)
			// serial := devices.Device_to_serial[_dev]
			// serialsList = append(serialsList, serial)
		}
	} else {
		devicesList = strings.Split(_devices_string, ",")
		// for _, dev := range devicesList {
		// 	serial := devices.Device_to_serial[dev]
		// 	serialsList = append(serialsList, serial)
		// }
	}

	return devicesList
}

func Get_Port(devicesList []string) [][2]int {
	var portsList [][2]int
	for _, device := range devicesList {
		// default uplink port and downlink port for each device
		ports := []int{devices.Device_to_port[device][0], devices.Device_to_port[device][1]}
		portsList = append(portsList, [2]int(ports))
	}
	return portsList
}

func Server_receive(stream quic.Stream, buf []byte) (float64, error) {
	_, err := stream.Read(buf)
	tsSeconds := binary.BigEndian.Uint32(buf[8:12])
	tsMicroseconds := binary.BigEndian.Uint32(buf[12:16])
	ts := float64(tsSeconds) + float64(tsMicroseconds)/1e9
	if err != nil {
		return -115, err
	}

	return ts, err
}

func Server_create_packet(euler uint32, pi uint32, datetimedec uint32, microsec uint32, seq uint32) []byte {
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

func Server_transmit(stream quic.Stream, message []byte) {
	_, err := stream.Write(message)
	if err != nil {
		log.Fatal(err)
	}
}
