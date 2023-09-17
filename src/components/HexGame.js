import {
    HexGrid,
    Layout,
    Hexagon,
    GridGenerator,
    Text
} from "react-hexgrid";

import {
    Heading, Box
} from '@chakra-ui/react';

import { useState } from "react";

const HexGame = (props) => {
    const [bluemoves, setBluemoves] = useState(true);
    // const [color, setColor] = useState('blue');
    const hexagons = GridGenerator.parallelogram(
        -6, 6, -6, 6
    );

    const onClick = (e, h) => {
        const hex = h.state.hex;
        console.log(e, h, hex, bluemoves);
        // setColor(bluemoves ? 'blue' : 'red');
        hex.color = bluemoves ? 'blue' : 'red';
        setBluemoves(!bluemoves);
    };

    return (
        <>
        <Box w='600px' h='600px'>
        <Heading>Hex grid</Heading>
            <HexGrid width='100%' height='100%'>
                <Layout size={{x: 3, y: 3}} flat={false}>
                    {
                        hexagons.map(
                            (hex, i) => <Hexagon
                                key={i}
                                q={hex.q}
                                r={hex.r}
                                s={hex.s}
                                onClick={onClick}
                                className={hex.color}
                            >
                                <Text fontSize="0.02em" fill="#000">
                                    {hex.q}, {hex.r}, {hex.s}
                                </Text>
                            </Hexagon>
                        )
                    }
                </Layout>
            </HexGrid>
        </Box>
        </>
    );
};

export default HexGame;
