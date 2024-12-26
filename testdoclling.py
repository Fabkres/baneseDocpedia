from docling.document_converter import DocumentConverter

source = "C:/Users/fabri/Documents/Artigos/DeepReinforcementLearningOfEventTriggeredCommunicationAndConsensusBasedControlForDistributedCooperativeTransport.pdf"  # Substitua pelo caminho do documento no seu sistema
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())
