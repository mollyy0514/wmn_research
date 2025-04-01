package wire

import (
	"bytes"
	"fmt"

	"github.com/mollyy0514/quic-go/internal/protocol"
	"github.com/mollyy0514/quic-go/quicvarint"
)

type FeedbackFrame struct {
	Feedback uint64
}

// TODO: parseFeedbackFrame function
func parseFeedbackFrame(r *bytes.Reader, _ protocol.Version) (*FeedbackFrame, error) {
	frame := &FeedbackFrame{}
	feedback, err := quicvarint.Read(r)
	frame.Feedback = feedback
	if err != nil {
		fmt.Println("ERR IN parseFeedbackFrame", err)
		return nil, err
	}
	return frame, nil
}

func (f *FeedbackFrame) Append(b []byte, _ protocol.Version) ([]byte, error) {
	b = append(b, feedbackFrameType)
	b = quicvarint.Append(b, f.Feedback)
	return b, nil
}

func (f *FeedbackFrame) Length(_ protocol.Version) protocol.ByteCount {
	return protocol.ByteCount(1 + quicvarint.Len(uint64(f.Feedback)))
}
