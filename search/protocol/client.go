package protocol

import (
  "fmt"
  "io"
  "time"
  "strings"
  "net/rpc"
  "bufio"
  "strconv"
  "os"
  "encoding/json"
  "log"
  "net"
  "image"
  "image/jpeg"
  "encoding/base64"
)

import (
  "github.com/henrycg/simplepir/pir"
  "github.com/henrycg/simplepir/matrix"
  "github.com/ahenzinger/underhood/underhood"
)

import (
  "github.com/ahenzinger/tiptoe/search/utils"
  "github.com/ahenzinger/tiptoe/search/corpus"
  "github.com/ahenzinger/tiptoe/search/config"
  "github.com/ahenzinger/tiptoe/search/embeddings"
  "github.com/ahenzinger/tiptoe/search/database"
)

import (
  myconfig "github.com/mora-2/simplepir/http/client/config"
	mypir "github.com/mora-2/simplepir/pir"
)

import (
  "github.com/fatih/color"
)


type UnderhoodAnswer struct {
  EmbAnswer underhood.HintAnswer
  UrlAnswer underhood.HintAnswer
}

type QueryType interface {
  bool | underhood.HintQuery | pir.Query[matrix.Elem64] | pir.Query[matrix.Elem32]
}

type AnsType interface {
  TiptoeHint | UnderhoodAnswer | pir.Answer[matrix.Elem64] | pir.Answer[matrix.Elem32]
}

type Client struct {
  params          corpus.Params

  embClient       *underhood.Client[matrix.Elem64]
  embInfo         *pir.DBInfo
  embMap          database.ClusterMap
  embIndices      map[uint64]bool

  urlClient       *underhood.Client[matrix.Elem32]
  urlInfo         *pir.DBInfo
  urlMap          database.SubclusterMap
  urlIndices      map[uint64]bool

  rpcClient       *rpc.Client
  useCoordinator  bool

  stepCount       int
}

var program string = "client.go"
var ip_config_file_path string = "/home/nsklab/yyh/similar/simplepir_law/http/client/config/ip_config.json"
//var ip_config_file_path string = "../data/offline/ip_config.json"
var offline_file_path string = "/home/nsklab/yyh/similar/simplepir_law/http/client/data/offline_data"
//var offline_file_path string = "../data/offline/offline_data"
var log_file_path string = "log.txt"

func NewClient(useCoordinator bool) *Client {
  c := new(Client)
  c.useCoordinator = useCoordinator
  return c
}

func (c *Client) Free() {
  c.urlClient.Free()
  c.embClient.Free()
}

func (c *Client) NumDocs() uint64 {
  return c.params.NumDocs
}

func (c *Client) NumClusters() int {
  if len(c.embMap) > 0 {
    return len(c.embMap)
  }
  return len(c.urlMap)
}

func (c *Client) printStep(text string) {
  col := color.New(color.FgGreen).Add(color.Bold)
  col.Printf("%d) %v\n", c.stepCount, text)
  c.stepCount += 1
}

func RunClient(coordinatorAddr string, conf *config.Config) {
  // ip config 
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
	//fmt.Printf("\rLoading...")
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
	fmt.Printf("Data loaded.\n")
  
  //tiptoe database
  //color.Yellow("Setting up client...")

  c := NewClient(true /* use coordinator */)
  //c.printStep("Getting metadata")
  hint := c.getHint(true /* keep conn */, coordinatorAddr)
  c.Setup(hint)
  logHintSize(hint)
  //cases := readIntListFromFile("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/rank_result/output.txt")
  cases := readIntListFromFile("../data/text_info/output.txt")
  facts := readLines("../data/text_info/facts.txt")
  //facts := readLines("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/rank_result/facts.txt")
  cols := make([][]uint, 0)
  for i:=0; i<1; i++{
    filePath := fmt.Sprintf("/home/nsklab/yyh/similar/tiptoe/search/mydata/law_data/1_cluster_%d.txt", i)
    //filePath := fmt.Sprintf("/home/nsklab/yyh/similar/tiptoe/search/mydata/data_paper/10w_datas/1_cluster_%d.txt", i)
    col0 := readintegers(filePath)
    cols = append(cols,col0)
  }
  in, out := embeddings.SetupEmbeddingProcess(c.NumClusters(), conf)
  //print("\nnumclusters:",c.NumClusters(),"\n")
  //col := color.New(color.FgYellow).Add(color.Bold)
  for {
    c.stepCount = 1
    //c.printStep("Running client preprocessing")
    perf := c.preprocessRound(coordinatorAddr, true /* verbose */, true /* keep conn */)
    //col.Printf("Enter private search query: ")
    text := utils.ReadLineFromStdin()
    fmt.Printf("\n\n")
    if (strings.TrimSpace(text) == "") || (strings.TrimSpace(text) == "quit") {
      break
    }
    //c.runRound0(cols,cases,facts,perf, in, out, text, coordinatorAddr, true /* verbose */, true /* keep conn */)
    c.runRound0(cols,decoder,offline_data,ip_addr,client_pir,cases,facts,perf, in, out, text, coordinatorAddr, true /* verbose */, true /* keep conn */)
  }

  if c.rpcClient != nil {
    c.rpcClient.Close()
  }
  in.Close()
  out.Close()
}

func (c *Client) preprocessRound(coordinatorAddr string, verbose, keepConn bool) Perf {
  var p Perf

  // Perform preprocessing
  start := time.Now()
  ct := c.PreprocessQuery()

  networkingStart := time.Now()
  offlineAns := c.applyHint(ct, keepConn, coordinatorAddr)
  p.tOffline, p.upOffline, p.downOffline = logOfflineStats(c.NumDocs(), networkingStart, ct, offlineAns)

  c.ProcessHintApply(offlineAns)
              
  p.clientPreproc = time.Since(start).Seconds()

  if verbose {
    fmt.Printf("\tPreprocessing complete -- %fs\n\n", p.clientPreproc)
  }

  return p
}


func (c *Client) runRound(p Perf, in io.WriteCloser, out io.ReadCloser, 
  text, coordinatorAddr string, verbose, keepConn bool) Perf {
  y := color.New(color.FgYellow, color.Bold)
  fmt.Printf("Executing query \"%s\"\n", y.Sprintf(text))

  // Build embeddings query
  start := time.Now()
  if verbose {
  c.printStep("Generating embedding of the query")
  }

  var query struct {
  Cluster_index uint64
  Emb           []int8
  }
  query.Cluster_index = 0
  query.Emb = readIntegersFromFile("/home/nsklab/yyh/similar/tiptoe/search/mydata/data/q_emb.txt",2)

  if query.Cluster_index >= uint64(c.NumClusters()) {
  panic("Should not happen")
  }

  if verbose {
  c.printStep(fmt.Sprintf("Building PIR query for cluster %d", query.Cluster_index))
  }

  embQuery := c.QueryEmbeddings(query.Emb, query.Cluster_index)
  p.clientSetup = time.Since(start).Seconds()

  // Send embeddings query to server
  if verbose {
  c.printStep("Sending SimplePIR query to server")
  }
  networkingStart := time.Now()
  embAns := c.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)
  p.t1, p.up1, p.down1 = logStats(c.params.NumDocs, networkingStart, embQuery, embAns)

  // Recover document and URL chunk to query for
  c.printStep("Decrypting server answer")
  embDec := c.ReconstructEmbeddingsWithinCluster(embAns, query.Cluster_index)
  scores := embeddings.SmoothResults(embDec, c.embInfo.P())

  //print("\nlen(score):",len(scores),"\n")
  docIndex := maxIndex(scores)
  indicesByScore := utils.SortByScores(scores)
  if verbose {
  fmt.Printf("\tDoc %d within cluster %d has the largest inner product with our query\n",
  docIndex, query.Cluster_index)
  c.printStep(fmt.Sprintf("Building PIR query for url/title of doc %d in cluster %d",
  docIndex, query.Cluster_index))
  }

  // Build URL query
  urlQuery, retrievedChunk := c.QueryUrls(query.Cluster_index, docIndex)

  // Send URL query to server
  if verbose {
  c.printStep(fmt.Sprintf("Sending PIR query to server for chunk %d", retrievedChunk))
  }
  networkingStart = time.Now()
  urlAns := c.getUrlsAnswer(urlQuery, keepConn, coordinatorAddr)
  p.t2, p.up2, p.down2 = logStats(c.params.NumDocs, networkingStart, urlQuery, urlAns)

  // Recover URLs of top 10 docs in chunk
  urls := c.ReconstructUrls(urlAns, query.Cluster_index, docIndex)
  if verbose {
  c.printStep("Reconstructed PIR answers.")
  fmt.Printf("\tThe top 10 retrieved urls are:\n")
  }

  j := 1
  for at := 0; at < len(indicesByScore); at++ {
  if scores[at] == 0 {
  break
  }

  doc := indicesByScore[at]
  _, chunk, index := c.urlMap.SubclusterToIndex(query.Cluster_index, doc)

  if chunk == retrievedChunk {
  if verbose {
  fmt.Printf("\t% 3d) [score %s] %s\n", j,
  color.YellowString(fmt.Sprintf("% 4d", scores[at])),
  color.BlueString(corpus.GetIthUrl(urls, index)))
  }
  j += 1
  if j > 10 {
  break
  }
  }
  }

  p.clientTotal = time.Since(start).Seconds()
  fmt.Printf("\tAnswered in:\n\t\t%v (preproc)\n\t\t%v (client)\n\t\t%v (round 1)\n\t\t%v (round 2)\n\t\t%v (total)\n---\n",
  p.clientPreproc, p.clientSetup, p.t1, p.t2, p.clientTotal)

  return p
}


//func (c *Client) runRound0(cols [][]uint,cases []int,facts []string,p Perf, in io.WriteCloser, out io.ReadCloser, 
  //text, coordinatorAddr string, verbose, keepConn bool)
func (c *Client) runRound0(cols [][]uint,decoder *json.Decoder,offline_data myconfig.Offline_data,ip_addr string,client_pir mypir.SimplePIR,cases []int,facts []string,p Perf, in io.WriteCloser, out io.ReadCloser, 
                          text, coordinatorAddr string, verbose, keepConn bool){
  //y := color.New(color.FgYellow, color.Bold)

  // Build embeddings query
  start := time.Now()
  //if verbose {
  //  c.printStep("Generating embedding of the query")
  //}

  var query struct {
    Cluster_index uint64
    Emb           []int8
  }
  io.WriteString(in, text + "\n")
  if err := json.NewDecoder(out).Decode(&query); err != nil { // get back embedding + cluster
    log.Printf("Did you remember to set up your python venv?")
    panic(err)
  }

  if query.Cluster_index >= uint64(c.NumClusters()) {
    panic("Should not happen")
  }

  //if verbose {
  //  c.printStep(fmt.Sprintf("Building PIR query for cluster %d", query.Cluster_index))
  //}

  embQuery := c.QueryEmbeddings(query.Emb, query.Cluster_index)
  
  p.clientSetup = time.Since(start).Seconds()

  // Send embeddings query to server
  //if verbose {
  //  c.printStep("Sending query to server")
  //}
  networkingStart := time.Now()
  embAns := c.getEmbeddingsAnswer(embQuery, true /* keep conn */, coordinatorAddr)
  p.t1, p.up1, p.down1 = logStats(c.params.NumDocs, networkingStart, embQuery, embAns)

  // Recover document and URL chunk to query for
  //c.printStep("Decrypting server answer")
  embDec := c.ReconstructEmbeddingsWithinCluster(embAns, query.Cluster_index)
  scores := embeddings.SmoothResults(embDec, c.embInfo.P())

  docIndex := maxIndex(scores)
  docIndex = uint64(cols[query.Cluster_index][docIndex])
  //print("\ndocIndex:",docIndex,"\n")
  //print("\ntime passed:",time.Since(start).Milliseconds(),"\n")
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
  //print("\ndocindex:",docIndex,"\n")
  //docIndex = uint64(cols[query.Cluster_index][docIndex]%55192)
  val = "近似法律文本:" + val + "\n"
  fmt.Println(val)
  //fact := facts[docIndex]
  //docIndex = uint64(cases[docIndex])
  //if verbose {
    //y.Printf("\tDoc %d has the largest inner product with our query\n",
               //docIndex)
    //print(fact,"\n")
  //}

}

func (c *Client) Setup(hint *TiptoeHint) {
  if hint == nil {
    panic("Hint is empty")
  }

  if hint.CParams.NumDocs == 0 {
    panic("Corpus is empty")
  }

  c.params = hint.CParams
  c.embInfo = &hint.EmbeddingsHint.Info
  c.urlInfo = &hint.UrlsHint.Info


  if hint.ServeEmbeddings {
    if hint.EmbeddingsHint.IsEmpty() {
      panic("Embeddings hint is empty")
    }

    c.embClient = utils.NewUnderhoodClient(&hint.EmbeddingsHint)

    c.embMap = hint.EmbeddingsIndexMap
    c.embIndices = make(map[uint64]bool)
    for _, v := range c.embMap {
      c.embIndices[v] = true
    }

    fmt.Printf("\tEmbeddings client: %s\n", utils.PrintParams(c.embInfo))
  }

  if hint.ServeUrls {
    if hint.UrlsHint.IsEmpty() {
      panic("Urls hint is empty")
    }
        
    c.urlClient = utils.NewUnderhoodClient(&hint.UrlsHint)

    c.urlMap = hint.UrlsIndexMap
    c.urlIndices = make(map[uint64]bool)
    for _, vals := range c.urlMap {
      for _, v := range vals {
        c.urlIndices[v.Index()] = true
      }
    }
 
    fmt.Printf("\tURL client: %s\n", utils.PrintParams(c.urlInfo))
  }

  if hint.ServeUrls && hint.ServeEmbeddings && 
     (len(c.urlMap) != len(c.embMap)) {
    fmt.Printf("Both maps don't have the same length: %d %d\n", len(c.urlMap), len(c.embMap))
//    panic("Both maps don't have same length.")
  }
}

func (c *Client) PreprocessQuery() *underhood.HintQuery {
  if c.params.NumDocs == 0 {
    panic("Not set up")
  }

  if c.embClient != nil {
    hintQuery := c.embClient.HintQuery()
    if c.urlClient != nil {
      c.urlClient.CopySecret(c.embClient)
    }
    return hintQuery
  } else if c.urlClient != nil {
    return c.urlClient.HintQuery()
  } else {
    panic("Should not happen")
  }
}

func (c *Client) ProcessHintApply(ans *UnderhoodAnswer) {
  if c.embClient != nil {
    c.embClient.HintRecover(&ans.EmbAnswer)
    c.embClient.PreprocessQueryLHE()
  }

  if c.urlClient != nil {
    c.urlClient.HintRecover(&ans.UrlAnswer)
    c.urlClient.PreprocessQuery()
  }
}

func (c *Client) QueryEmbeddings(emb []int8, clusterIndex uint64) *pir.Query[matrix.Elem64] {
  if c.params.NumDocs == 0 {
    panic("Not set up")
  }

  dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
  m := c.embInfo.M
  dim := uint64(len(emb))

  if m % dim != 0 {
    panic("Should not happen")
  }
  if dbIndex % dim != 0 {
    panic("Should not happen")
  }

  _, colIndex := database.Decompose(dbIndex, m)
  colIndex = clusterIndex * 192
  arr := matrix.Zeros[matrix.Elem64](m, 1)
  for j := uint64(0); j < dim; j++ {
    arr.AddAt(colIndex + j, 0, matrix.Elem64(emb[j]))
  }

  return c.embClient.QueryLHE(arr)
}

func (c *Client) QueryEmbeddings0(emb []int8, clusterIndex uint64) *pir.Query[matrix.Elem64] {
  if c.params.NumDocs == 0 {
    panic("Not set up")
  }

  dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
  m := c.embInfo.M
  dim := uint64(len(emb))

  if m % dim != 0 {
    panic("Should not happen")
  }
  if dbIndex % dim != 0 {
    panic("Should not happen")
  }

  _, colIndex := database.Decompose(dbIndex, m)
  colIndex = colIndex * 192 * clusterIndex
  arr := matrix.Zeros[matrix.Elem64](m, 1)
  for j := uint64(0); j < dim; j++ {
    arr.AddAt(j, 0, matrix.Elem64(emb[j]))
  }

  return c.embClient.QueryLHE(arr)
}

func (c *Client) QueryUrls(clusterIndex, docIndex uint64) (*pir.Query[matrix.Elem32], uint64) {
  if c.params.NumDocs == 0 {
    panic("Not set up")
  }

  dbIndex, chunkIndex, _ := c.urlMap.SubclusterToIndex(clusterIndex, docIndex) 

  return c.urlClient.Query(dbIndex), chunkIndex
}

func (c *Client) ReconstructEmbeddings(answer *pir.Answer[matrix.Elem64], 
                                       clusterIndex uint64) uint64 {
  vals := c.embClient.RecoverLHE(answer)
  dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
  rowIndex, _ := database.Decompose(dbIndex, c.embInfo.M)
  print("\nrowindex:",rowIndex,"\n")
  res := vals.Get(rowIndex, 0)

  return uint64(res)
}

func (c *Client) checkans(answer *pir.Answer[matrix.Elem64], 
  clusterIndex, p uint64, corp *corpus.Corpus, emb []int8){
  vals := c.embClient.RecoverLHE(answer)
  for i:= uint64(0); i < uint64(len(vals.Data())); i++ {
    dbIndex := c.embMap.ClusterToIndex(uint(i))
    rowIndex, _ := database.Decompose(dbIndex, c.embInfo.M)
    res := vals.Get(rowIndex, 0)
    docEmb := corp.GetEmbedding(uint64(corp.ClusterToIndex(uint(i))))
    shouldBe := embeddings.InnerProduct(docEmb, emb)
    if int(res) != shouldBe {
      fmt.Printf("Recovering doc %d: got %d instead of %d\n",
                 i, res, shouldBe)
      panic("Bad answer")
    }
  }
}

func (c *Client) ReconstructEmbeddingsWithinCluster(answer *pir.Answer[matrix.Elem64], 
                                                    clusterIndex uint64) []uint64 {
  //dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
  //rowStart, _ := database.Decompose(dbIndex, c.embInfo.M)
  //rowEnd := database.FindEnd(c.embIndices, rowStart, colIndex,
                             //c.embInfo.M, c.embInfo.L, 0)
    
  vals := c.embClient.RecoverLHE(answer)
  res := make([]uint64, len(vals.Data()))
  at := 0
  //print("\nlen(vals.Data()):",len(vals.Data()),"\n")
  for j := uint64(0); j < uint64(len(vals.Data())); j++ {
    //dbIndex := c.embMap.ClusterToIndex(uint(j))
    //rowIndex, _ := database.Decompose(dbIndex, c.embInfo.M)
    res[at] = uint64(vals.Get(j, 0))
    at += 1
  }
  return res
}


func (c *Client) ReconstructEmbeddingsWithinCluster0(answer *pir.Answer[matrix.Elem64], 
  clusterIndex uint64) []uint64 {
  //dbIndex := c.embMap.ClusterToIndex(uint(clusterIndex))
  //rowStart, _ := database.Decompose(dbIndex, c.embInfo.M)
  //rowEnd := database.FindEnd(c.embIndices, rowStart, colIndex,
  //c.embInfo.M, c.embInfo.L, 0)

  vals := c.embClient.RecoverLHE(answer)
  res := make([]uint64, len(vals.Data()))
  at := 0
  print("\nlen(vals.Data()):",len(vals.Data()),"\n")
  for j := uint64(0); j < uint64(len(vals.Data())); j++ {
    dbIndex := c.embMap.ClusterToIndex(uint(j))
    rowIndex, _ := database.Decompose(dbIndex, c.embInfo.M)
    res[at] = uint64(vals.Get(rowIndex, 0))
    at += 1
  }
  return res
}

func (c *Client) ReconstructUrls(answer *pir.Answer[matrix.Elem32], 
                                 clusterIndex, docIndex uint64) string {
  dbIndex, _, _ := c.urlMap.SubclusterToIndex(clusterIndex, docIndex)
  rowStart, colIndex := database.Decompose(dbIndex, c.urlInfo.M)
  rowEnd := database.FindEnd(c.urlIndices, rowStart, colIndex, 
                             c.urlInfo.M, c.urlInfo.L, c.params.UrlBytes)

  vals := c.urlClient.Recover(answer)

  out := make([]byte, rowEnd - rowStart)
  for i, e := range vals[rowStart:rowEnd] {
    out[i] = byte(e)
  }

  if c.params.CompressUrl {
    res, err := corpus.Decompress(out)
    for ; err != nil; {
      out = out[:len(out)-1]
      if len(out) == 0 {
        panic("Should not happen")
      }
      res, err = corpus.Decompress(out)
    }
    return strings.TrimRight(res, "\x00")
  }

  return strings.TrimRight(string(out), "\x00")
}

func makeRPC[Q QueryType, A AnsType](query *Q, reply *A, useCoordinator, keepConn bool, 
                                     tcp, rpc string, client *rpc.Client) *rpc.Client {
  if !useCoordinator {
    conn := utils.DialTCP(tcp)
    utils.CallTCP(conn, "Server." + rpc, query, reply)
    conn.Close()
  } else {
    if client == nil {
      client = utils.DialTLS(tcp)
    }

    utils.CallTLS(client, "Coordinator." + rpc, query, reply)

    if !keepConn {
      client.Close()
      client = nil
    }
  }

  return client
}

func (c *Client) getHint(keepConn bool, tcp string) *TiptoeHint {
  query := true
  hint := TiptoeHint{}
  c.rpcClient = makeRPC[bool, TiptoeHint](&query, &hint, c.useCoordinator, keepConn, 
                                          tcp, "GetHint", c.rpcClient)
  return &hint
}

func (c *Client) applyHint(ct *underhood.HintQuery, 
                                     keepConn bool, 
				     tcp string) *UnderhoodAnswer {
  ans := UnderhoodAnswer{}
  c.rpcClient = makeRPC[underhood.HintQuery, UnderhoodAnswer](ct, &ans,
                                                               c.useCoordinator, keepConn,
							       tcp, "ApplyHint",
							       c.rpcClient)
  return &ans
}

func (c *Client) getEmbeddingsAnswer(query *pir.Query[matrix.Elem64], 
                                     keepConn bool, 
	  		             tcp string) *pir.Answer[matrix.Elem64] {
  ans := pir.Answer[matrix.Elem64]{}
  c.rpcClient = makeRPC[pir.Query[matrix.Elem64], pir.Answer[matrix.Elem64]](query, &ans, 
                                                                             c.useCoordinator, keepConn, 
									     tcp, "GetEmbeddingsAnswer",
								             c.rpcClient)
  return &ans
}

func (c *Client) getUrlsAnswer(query *pir.Query[matrix.Elem32], 
                               keepConn bool, 
			       tcp string) *pir.Answer[matrix.Elem32] {
  ans := pir.Answer[matrix.Elem32]{}
  c.rpcClient = makeRPC[pir.Query[matrix.Elem32], pir.Answer[matrix.Elem32]](query, &ans, 
                                                                             c.useCoordinator, keepConn, 
									     tcp, "GetUrlsAnswer",
								             c.rpcClient)
  return &ans
}

func (c *Client) closeConn() {
  if c.rpcClient != nil {
    c.rpcClient.Close()
    c.rpcClient = nil
  }
}

func readIntListFromFile(filename string) []int {
	file, err := os.Open(filename)
	if err != nil {
		return nil
	}
	defer file.Close()
	var intList []int
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		num, err := strconv.Atoi(scanner.Text())
		if err != nil {
			return nil
		}
		intList = append(intList, num)
	}
	if err := scanner.Err(); err != nil {
		return nil
	}
	return intList
}

func readLines(filename string) []string {
  file, err := os.Open(filename)
  if err != nil {
      return nil
  }
  defer file.Close()

  var lines []string
  scanner := bufio.NewScanner(file)
  for scanner.Scan() {
      lines = append(lines, scanner.Text())
  }
  if err := scanner.Err(); err != nil {
      return nil
  }

  return lines
}

func readIntegersFromFile(filePath string, linenumber int) ([]int8) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var lineCount int
	for scanner.Scan() {
		lineCount++
		if lineCount == linenumber {
			line := scanner.Text()
			fields := strings.Fields(line)

			var integers []int8

			for _, field := range fields {
				num, err := strconv.Atoi(field)
				if err != nil {
					return nil
				}
				integers = append(integers, int8(num))
			}
			return integers
		}
	}

	if err := scanner.Err(); err != nil {
		return nil
	}
	return nil
}

func readintegers(filename string) []uint{
  file, err := os.Open(filename)
  if err != nil {
      fmt.Println("Error opening file:", err)
      return nil
  }
  defer file.Close() 

  var numbers []uint 

  scanner := bufio.NewScanner(file)
  for scanner.Scan() { // 逐行读取
      line := scanner.Text() // 获取行文本
      number, err := strconv.Atoi(line) // 将行文本转换为整数
      if err != nil {
          fmt.Println("Error converting string to int:", err)
          continue // 遇到错误时跳过这一行
      }
      numbers = append(numbers, uint(number)) // 将整数添加到切片中
  }

  // 检查是否有扫描时发生的错误
  if err := scanner.Err(); err != nil {
      fmt.Println("Error reading file:", err)
  }

  return numbers // 打印所有读取的整数
}

func maxIndex(nums []int) uint64 {
	if len(nums) == 0 {
    print("\nbad")
		return 0 // 返回 -1 表示列表为空
	}

	maxIdx := uint64(0) // 假设列表中第一个元素为最大值的索引
	maxVal := nums[0]
  for j := uint64(0); j < uint64(len(nums)); j++ {
    if nums[j] > maxVal{
      maxVal = nums[j]
      maxIdx = j
    }
  }
	return maxIdx
}

func storeimagefromstr(compressedImageStr string,index uint64){
  imgData, err := base64.StdEncoding.DecodeString(compressedImageStr)
  if err != nil {
      fmt.Println("Error decoding base64 string:", err)
      return
  }

  // 从字节数组创建图像
  img, _, err := image.Decode(strings.NewReader(string(imgData)))
  if err != nil {
      fmt.Println("Error decoding image:", err)
      return
  }

  // 保存还原的图像到文件
  path := "/home/nsklab/yyh/similar/tiptoe/search/data/images_result/" + strconv.Itoa(int(index)) + ".jpg"
  outputfile, err := os.Create(path)
  if err != nil {
      fmt.Println("Error creating output file:", err)
      return
  }
  defer outputfile.Close()

  // 将图像保存为 JPEG 格式
  err = jpeg.Encode(outputfile, img, nil)
  if err != nil {
      fmt.Println("Error encoding image:", err)
      return
  }

  fmt.Println(path)
}