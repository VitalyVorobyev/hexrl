import './App.css';
import { ChakraProvider } from '@chakra-ui/react';
import HexGame from './components/HexGame';

function App() {
  return (
    <ChakraProvider>
      <HexGame />
    </ChakraProvider>
  );
}

export default App;
