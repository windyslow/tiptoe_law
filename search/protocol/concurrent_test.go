package protocol

import (
  "fmt"
  "time"
  "bufio"
  "strconv"
  "os"
  "encoding/json"
  "log"
  "net"
  "testing"
  "sync"
)


import (
  "github.com/ahenzinger/tiptoe/search/embeddings"
)


import (
	"github.com/henrycg/simplepir/pir"
	"github.com/henrycg/simplepir/matrix"
)

import (
  myconfig "github.com/mora-2/simplepir/http/client/config"
	mypir "github.com/mora-2/simplepir/pir"
)

import (
  "github.com/fatih/color"
)



func TestCoooooF(t *testing.T) {
	// ip config 
	coordinatorAddr := "219.245.186.51:1237"
	ip_file, err := os.Open(ip_config_file_path)
	if err != nil {
		fmt.Println("Error loading ip_config.json:", err.Error())
	}
	defer ip_file.Close()

	var ip_cfg myconfig.IP_Conn
	decoder := json.NewDecoder(ip_file)
	err = decoder.Decode(&ip_cfg)
	if err != nil {
		fmt.Println("Error decoding ip_config:", err.Error())
	}

	ip_addr := ip_cfg.IpAddr + ":" + fmt.Sprint(ip_cfg.OnlinePort)
//create log file
	logFile, err := os.OpenFile(log_file_path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal("Cannot create log file: ", err.Error())
	}
	defer logFile.Close()
	log.SetOutput(logFile)
/*--------------pre loading start-------------*/
	fmt.Printf("\rLoading...")
	offline_file, err := os.Open(offline_file_path)
	if err != nil {
		fmt.Println("Error opening offline_data file:", err.Error())
	}
	defer offline_file.Close()

	var offline_data myconfig.Offline_data
	decoder = json.NewDecoder(offline_file)
	err = decoder.Decode(&offline_data)
	if err != nil {
		fmt.Println("Error loading offline_data:", err.Error())
	}
	// create client_pir
	client_pir := mypir.SimplePIR{}
		fmt.Printf("\rData loaded.\n")

	//tiptoe database
	color.Yellow("Setting up client...")

	c := NewClient(true /* use coordinator */)
	//c.printStep("Getting metadata")
	//hint := c.getHint(true /* keep conn */, coordinatorAddr)
	//c.Setup(hint)
	//logHintSize(hint)
	//var wg sync.WaitGroup
	
	
	var clients [64]*Client
	var emb_querys	[64]*pir.Query[matrix.Elem64]
	for i:=0;i<len(clients);i++{
		clients[i] = NewClient(true /* use coordinator */)
	}
	for i:=0;i<len(clients);i++ {
		//wg.Add(1)
		emb_querys[i] = clients[i].cycle(decoder,offline_data,ip_addr,client_pir,coordinatorAddr, true /* verbose */, true /* keep conn */)
	}

	fmt.Printf("begin test\n")
	startTime := time.Now()
	var wg sync.WaitGroup	
	for i:=0;i<len(clients);i++{
		wg.Add(1)
		go clients[i].runRound001(coordinatorAddr,emb_querys[i],&wg)
	}
	wg.Wait()
	duration := time.Since(startTime)
	fmt.Printf("程序运行时间: %dms\n", int(duration.Milliseconds()))
	if c.rpcClient != nil {
		c.rpcClient.Close()
	}
}

func (c *Client) cycle(decoder *json.Decoder,offline_data myconfig.Offline_data,ip_addr string,client_pir mypir.SimplePIR, 
	coordinatorAddr string, verbose, keepConn bool)*pir.Query[matrix.Elem64]{
	//defer wg.Done()
	hint := c.getHint(true /* keep conn */, coordinatorAddr)
	c.printStep("Getting metadata")
	c.Setup(hint)
	logHintSize(hint)
	c.printStep("Running client preprocessing")
	perf := c.preprocessRound(coordinatorAddr, true /* verbose */, true /* keep conn */)
	fmt.Printf("\n\n")
	//c.runRound0(cols,cases,facts,perf, in, out, text, coordinatorAddr, true /* verbose */, true /* keep conn */)
	return c.runRound000(decoder,offline_data,ip_addr,client_pir,perf, coordinatorAddr, true /* verbose */, true /* keep conn */)
}

func (c *Client) runRound000(decoder *json.Decoder,offline_data myconfig.Offline_data,ip_addr string,client_pir mypir.SimplePIR,p Perf, 
	coordinatorAddr string, verbose, keepConn bool)*pir.Query[matrix.Elem64]{
	//y := color.New(color.FgYellow, color.Bold)

	// Build embeddings query
	start := time.Now()
	if verbose {
	c.printStep("Generating embedding of the query")
	}

	var query struct {
	Cluster_index uint64
	Emb           []int8
	}
	query.Emb = readintegers0("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/image_bytes/test_embedding.txt")
	query.Cluster_index = 0

	embQuery := c.QueryEmbeddings(query.Emb, query.Cluster_index)
	print("\ncluster_index:",query.Cluster_index,"\n")
	p.clientSetup = time.Since(start).Seconds()
	// Send embeddings query to server
	if verbose {
	c.printStep("Sending query to server")
	}
	return embQuery
}

func (c *Client) runRound001(coordinatorAddr string, embQuery *pir.Query[matrix.Elem64], wg *sync.WaitGroup)*pir.Answer[matrix.Elem64]{
	defer wg.Done()
	embAns := c.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)
	return embAns
}

func (c *Client) runRound00(decoder *json.Decoder,offline_data myconfig.Offline_data,ip_addr string,client_pir mypir.SimplePIR,p Perf, 
	coordinatorAddr string, verbose, keepConn bool){
	//y := color.New(color.FgYellow, color.Bold)

	// Build embeddings query
	start := time.Now()
	if verbose {
	c.printStep("Generating embedding of the query")
	}

	var query struct {
	Cluster_index uint64
	Emb           []int8
	}
	query.Emb = readintegers0("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/image_bytes/test_embedding.txt")
	query.Cluster_index = 0

	embQuery := c.QueryEmbeddings(query.Emb, query.Cluster_index)
	print("\ncluster_index:",query.Cluster_index,"\n")
	p.clientSetup = time.Since(start).Seconds()
	// Send embeddings query to server
	if verbose {
	c.printStep("Sending query to server")
	}
	networkingStart := time.Now()


	embAns := c.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)


	p.t1, p.up1, p.down1 = logStats(c.params.NumDocs, networkingStart, embQuery, embAns)

	// Recover document and URL chunk to query for
	c.printStep("Decrypting server answer")
	embDec := c.ReconstructEmbeddingsWithinCluster(embAns, query.Cluster_index)
	scores := embeddings.SmoothResults(embDec, c.embInfo.P())

	docIndex := uint64(maxIndex(scores))
	// build query
	var client_state []mypir.State
	var query0 mypir.MsgSlice
	cs, q := client_pir.Query(uint64(docIndex), offline_data.Shared_state, offline_data.P, offline_data.Info)
	client_state = append(client_state, cs)
	query0.Data = append(query0.Data, q)
	conn, err := net.Dial("tcp", ip_addr)
	if err != nil {
	fmt.Println("Error connecting:", err.Error())
	return
	}
	defer conn.Close()
	encoder := json.NewEncoder(conn)
	err = encoder.Encode(query0)
	if err != nil {
	fmt.Println("Error encoding query:", err.Error())
	return
	}

	//log out
	log.Printf("[%v][%v][1. Send built query]\t Elapsed:%v \tSize:%vKB", program, conn.LocalAddr(),
	mypir.PrintTime(start), float64(query0.Size()*uint64(offline_data.P.Logq)/(8.0*1024.0)))
	//receive answer
	var answer mypir.Msg
	decoder = json.NewDecoder(conn)
	err = decoder.Decode(&answer)
	if err != nil {
	fmt.Println("Error decoding answer:", err.Error())
	return
	}
	//log out
	log.Printf("[%v][%v][2. Receive answer]\t Elapsed:%v \tSize:%vKB", program, conn.LocalAddr(),
	mypir.PrintTime(start), float64(answer.Size()*uint64(offline_data.P.Logq)/(8.0*1024.0)))

	//resconstruction
	val := client_pir.StrRecover(uint64(docIndex), uint64(0), offline_data.Offline_download,
	query0.Data[0], answer, offline_data.Shared_state,
	client_state[0], offline_data.P, offline_data.Info)
	val = val
	//print("\ndocindex:",docIndex,"\n")
	//docIndex = uint64(cols[query.Cluster_index][docIndex]%55192)
	//fact := facts[docIndex]
	//docIndex = uint64(cases[docIndex])
	//if verbose {
	//y.Printf("\tDoc %d has the largest inner product with our query\n",
	//docIndex)
	//print(fact,"\n")
	//}

}

func readintegers0(filename string) []int8{
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close() 
  
	var numbers []int8 
  
	scanner := bufio.NewScanner(file)
	for scanner.Scan() { // 逐行读取
		line := scanner.Text() // 获取行文本
		number, err := strconv.Atoi(line) // 将行文本转换为整数
		if err != nil {
			fmt.Println("Error converting string to int:", err)
			continue // 遇到错误时跳过这一行
		}
		numbers = append(numbers, int8(number)) // 将整数添加到切片中
	}
  
	// 检查是否有扫描时发生的错误
	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
	}
  
	return numbers // 打印所有读取的整数
}